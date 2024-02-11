"""
Copyright 2024 Kina
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import asyncio
import io
import json
import re
import tempfile
import uuid
from collections.abc import Iterable

import click
import librosa
import numpy as np
import numpy.ma as ma
import pyworld
from scipy.interpolate import Akima1DInterpolator
import soundfile
from voicevox import Client

from kodama.yomi2voca import yomi2voca

phoneme_pattern = re.compile(r"^\[\s*(\d+)\s*(\d+)]\s*[\d\-.]+\s*(.+?)\r?$", flags=re.MULTILINE)


def make_words(voca: str, silence_at_ends: bool = True) -> list[str]:
    words = [
        'silB',
        voca,
        'silE'
    ] if silence_at_ends else [voca]

    return words


def make_gram_dict(words: Iterable[str]) -> str:
    result = []
    for i, word in enumerate(words):
        result.append(f"{i} [w_{i}] {word}")

    return '\n'.join(result)


def make_sequential_gram_dfa(count: int) -> str:
    return '\n'.join(
        [f"{i} {count - i - 1} {i + 1} 0 {1 if i == 0 else 1}" for i in range(count)]
        + [f"{count} -1 -1 1 0"]
    )


async def phoneme_segmentation(utterance, julius_executable, hmmdefs, gram_dfa, gram_dict):
    with (
        tempfile.NamedTemporaryFile(delete=True, suffix=".dfa", delete_on_close=False) as dfa_file,
        tempfile.NamedTemporaryFile(delete=True, suffix=".dict", delete_on_close=False) as dict_file
    ):
        dfa_file.write(gram_dfa.encode())
        dfa_file.close()
        dict_file.write(gram_dict.encode())
        dict_file.close()
        proc = await asyncio.create_subprocess_shell(
            f"""echo {utterance} |\
                {julius_executable}\
                 -h {hmmdefs}\
                  -dfa {dfa_file.name}\
                   -v {dict_file.name}\
                    -palign\
                     -input file -charconv sjis utf-8""",
            shell=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        result = None
        while result is None:
            line = await proc.stdout.readline()
            print(line.decode('shift_jis'))
            if b'search failed' in line:
                raise ValueError()
            if b'begin forced alignment' in line:
                result = (await proc.stdout.readuntil(b'end forced alignment')).decode("shift_jis")

        proc.terminate()

        segmentation_matches = phoneme_pattern.finditer(result)

    return [
        (int(m.group(1)), int(m.group(2)), m.group(3))
        for m in segmentation_matches
    ]


def is_vowel(phoneme: str) -> bool:
    return phoneme in ['a', 'e', 'i', 'u', 'o', 'N']


def interpf0(f0):
    v = np.arange(f0.size)[~f0.mask]
    f0 = Akima1DInterpolator(v, f0[~f0.mask])(np.arange(f0.size))
    f0[:v[0]] = f0[v[0]]
    f0[v[-1] + 1:] = f0[v[-1]]
    return f0


def align_array(f0, f0_):
    f0 = ma.masked_equal(f0, 0.)
    f0 = (f0 - np.mean(f0)) + np.mean(f0)
    if f0_.size > f0.size:
        f0 = np.pad(f0, (0, f0_.size - f0.size), mode="edge")
        f0_ = f0_[:f0.size]
    else:
        f0 = f0[:f0_.size]
        f0_ = np.pad(f0_, (0, f0.size - f0_.size), mode="edge")
    modifier = np.copy(f0_)
    modifier[modifier > 0] = 1
    f0 = f0 * modifier
    f0 = ma.masked_equal(f0, 0.)
    f0 = f0 - np.mean(f0) + np.mean(ma.masked_equal(f0_, 0.))
    f0 = interpf0(f0)
    modifier = np.copy(f0_)
    modifier[modifier > 0] = 1
    f0 = f0 * modifier
    return f0


def vowel_mask(frame_segmentation, size):
    result = np.ones(size)
    for s, e, p in frame_segmentation:
        if is_vowel(p):
            result[s:e + 1] = 0

    return result


def make_vvproj(audio_query, speaker, transcript):
    key = str(uuid.uuid4())
    return {
        "appVersion": "0.14.11",
        "audioKeys": [key],
        "audioItems": {
            key: {
                "text": transcript,
                "engineId": "074fc39e-678b-4c13-8916-ffca8d505d1d",
                "styleId": int(speaker),
                "query": {
                    "accentPhrases": [
                        {
                            "moras": [(
                                {
                                    "text": mora.text,
                                    "consonant": mora.consonant,
                                    "consonantLength": mora.consonant_length,
                                    "vowel": mora.vowel,
                                    "vowelLength": mora.vowel_length,
                                    "pitch": mora.pitch,
                                } if mora.consonant is not None else {
                                    "text": mora.text,
                                    "vowel": mora.vowel,
                                    "vowelLength": mora.vowel_length,
                                    "pitch": mora.pitch,
                                }
                            ) for mora in accent_phrase.moras],
                            "accent": accent_phrase.accent,
                            "isInterrogative": accent_phrase.is_interrogative,
                        } for accent_phrase in audio_query.accent_phrases
                    ],
                    "speedScale": audio_query.speed_scale,
                    "pitchScale": audio_query.pitch_scale,
                    "intonationScale": audio_query.intonation_scale,
                    "volumeScale": audio_query.volume_scale,
                    "prePhonemeLength": audio_query.pre_phoneme_length,
                    "postPhonemeLength": audio_query.post_phoneme_length,
                    "outputSamplingRate": audio_query.output_sampling_rate,
                    "outputStereo": audio_query.output_stereo,
                    "kana": audio_query.kana,
                }
            }
        }
    }


async def main(julius_executable,
               hmmdefs,
               adinrec_executable,
               transcript,
               speaker,
               utterance,
               outfile,
               labfile,
               outtype,
               base_pitch,
               hz_to_pitch):
    proc = await asyncio.create_subprocess_shell(f"{adinrec_executable} {utterance}", shell=True,
                                                 stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.PIPE)
    print("waiting record instance...")
    await proc.stderr.readuntil(b'please speak')
    print("please speak")

    stdout, stderr = await proc.communicate()
    print(stderr.decode())

    x, fs = soundfile.read(utterance)

    async with Client() as client:
        audio_query = await client.create_audio_query(
            transcript, speaker=speaker
        )

        morae = [mora
                 for accent_phrase in audio_query.accent_phrases
                 for mora in accent_phrase.moras + ([accent_phrase.pause_mora] if accent_phrase.pause_mora else [])
                 ]

        print([yomi2voca(mora.text) for mora in morae])

        voca = ' '.join([yomi2voca(mora.text) for mora in morae])
        words = make_words(voca, silence_at_ends=True)
        gram_dict = make_gram_dict(words)
        gram_dfa = make_sequential_gram_dfa(len(words))
        print(gram_dfa)
        print(gram_dict)

        frame_segmentation = await phoneme_segmentation(utterance, julius_executable, hmmdefs, gram_dfa, gram_dict)

        f0, sp, ap = pyworld.wav2world(x, fs, frame_period=10.)
        f0 = ma.masked_equal(f0, 0.)

        f0 = ma.array(f0, mask=vowel_mask(frame_segmentation, f0.shape))

        f0 = interpf0(f0)

        f0 = ma.array(f0, mask=vowel_mask(frame_segmentation, f0.shape))

        zscores = f0 - np.mean(f0)

        second_segmentation = [
            (begin * 0.01, (end + 1) * 0.01) for begin, end, _ in frame_segmentation
        ]

        phoneme_lengths = [e - s for s, e in second_segmentation]
        zs = [np.mean(zscores[s:e]) for s, e, _ in frame_segmentation]

        phoneme_lengths_iter = iter(phoneme_lengths)
        z_iter = iter(zs)

        # omit silence
        audio_query.pre_phoneme_length = next(phoneme_lengths_iter)
        next(z_iter)
        for mora in morae:
            if mora.consonant:
                mora.consonant_length = next(phoneme_lengths_iter)
                next(z_iter)

            z = next(z_iter)
            if ma.is_masked(z):
                z = 0

            mora.vowel_length = next(phoneme_lengths_iter)
            mora.pitch = base_pitch + z * hz_to_pitch

        audio_query.post_phoneme_length = next(phoneme_lengths_iter)

        try:
            resp = await audio_query.synthesis(speaker=8)
        except Exception as e:
            print(e)

    if outtype == "force_pitch":
        x_, fs_ = soundfile.read(io.BytesIO(resp))
        x = librosa.resample(x, orig_sr=fs, target_sr=fs_)
        f0_, sp_, ap_ = pyworld.wav2world(x_, fs_, frame_period=5.)
        f0, _, _ = pyworld.wav2world(x, fs_, frame_period=5.)
        f0 = align_array(f0, f0_)

        resp = pyworld.synthesize(f0, sp_, ap_, 24000, frame_period=5.)

        soundfile.write(outfile, resp, samplerate=24000)
    elif outtype == "vvproj":
        with open(outfile, mode="w") as f:
            json.dump(make_vvproj(audio_query, speaker, transcript), f)
    else:
        with open(outfile, mode="wb") as f:
            f.write(resp)

    if labfile is not None:
        with open(labfile, "wb") as f:
            f.write("\n".join([
                f"{s:.7f} {e:.7f} {p}" for s, e, p in frame_segmentation
            ]).encode())


@click.command()
@click.option("-j", "--julius", "julius_executable", required=True, type=click.Path(exists=True, executable=True),
              help="Path to the julius")
@click.option("-h", "--hmmdefs", "hmmdefs", required=True, type=click.Path(exists=True, readable=True),
              help="Path to the hmmdefs")
@click.option("-a", "--adinrec", "adinrec_executable", required=True, type=click.Path(exists=True, executable=True),
              help="Path to the adinrec executable")
@click.option("-t", "--transcript", "transcript", required=True, prompt=True, type=str, help="Transcript")
@click.option("-s", "--speaker", "speaker", required=True, type=str, help="Speaker ID")
@click.option("-r", "--record-utterance", "utterance", required=True, type=str, help="Utterance Record File")
@click.option("-o", "--out", "outfile", required=False, type=str, help="Output file")
@click.option("-l", "--lab", "labfile", required=False, type=str, help="Output Lab file")
@click.option('--vvproj', 'outtype', flag_value="vvproj", is_flag=True, help="Output as Voicevox Project File")
@click.option("-f", "--force-pitch", "outtype", flag_value="force_pitch", is_flag=True,
              help="Force pitch using WORLD Vocoder")
@click.option("-b", "--base-pitch", "base_pitch", default=5.775, show_default=True, type=float, help="Base pitch")
@click.option("-z", "--hz-to-pitch", "hz_to_pitch", default=0.00625, show_default=True, type=float, help="Hz to pitch")
def command(julius_executable,
            hmmdefs,
            adinrec_executable,
            transcript,
            speaker,
            utterance,
            outfile,
            labfile,
            outtype,
            base_pitch,
            hz_to_pitch):
    asyncio.run(
        main(julius_executable, hmmdefs, adinrec_executable, transcript, speaker, utterance, outfile, labfile, outtype,
             base_pitch, hz_to_pitch), debug=True)