import asyncio
import io
import itertools
import math
import os
import pathlib
import re
import tempfile
from pprint import pprint

import librosa

from asyncio import StreamWriter, StreamReader
from collections.abc import Collection, Iterable
import pandas as pd
import numpy as np
import numpy.ma as ma
import pyworld
import scipy.stats
import soundfile

from voicevox import Client

from yomi2voca import yomi2voca


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


def is_vowel(phoneme: str) -> bool:
    return phoneme in ['a', 'e', 'i', 'u', 'o', 'N']


async def main():
    async with Client() as client:
        audio_query = await client.create_audio_query(
            "本日は晴天なり", speaker=8
        )
        pprint(audio_query.to_dict())

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

        with (
            tempfile.NamedTemporaryFile(delete=True, suffix=".dfa", delete_on_close=False) as dfa_file,
            tempfile.NamedTemporaryFile(delete=True, suffix=".dict", delete_on_close=False) as dict_file
        ):
            # temp_dir = pathlib.Path("./wav")
            utterance = "./rawvoice.wav"
            dfa_file.write(gram_dfa.encode())
            dfa_file.close()
            dict_file.write(gram_dict.encode())
            dict_file.close()
            proc = await asyncio.create_subprocess_shell(
                f"""echo {utterance} |\
                julius.exe\
                 -h hmmdefs_monof_mix16_gid.binhmm\
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
                if b'begin forced alignment' in line:
                    result = (await proc.stdout.readuntil(b'end forced alignment')).decode("shift_jis")

            proc.terminate()

            # path = pathlib.Path(temp_dir)
            # utterance = max(path.iterdir(), key=os.path.getmtime)
            # print(utterance)

            # x, fs = librosa.load(utterance, sr=24000, dtype=numpy.float64)
            x, fs = soundfile.read(utterance)

        f0, sp, _ = pyworld.wav2world(x, fs, frame_period=10.)
        f0 = ma.masked_equal(f0, 0.)
        zscores = f0 - np.mean(f0)
        process_morae = morae[:]
        phoneme_pattern = re.compile(r"^\[\s*(\d+)\s*(\d+)]\s*[\d\-.]+\s*(.+)$", flags=re.MULTILINE)

        segmentation_matches = phoneme_pattern.findall(result)
        frame_segmentation = [
            (int(begin), int(end))
            for begin, end, _ in segmentation_matches
        ]
        second_segmentation = [
            (begin * 0.01, (end + 1) * 0.01) for begin, end in frame_segmentation
        ]

        phoneme_lengths = [e - s for s, e in second_segmentation]
        zs = [np.mean(zscores[s:e]) for s, e in frame_segmentation]

        phoneme_lengths_iter = iter(phoneme_lengths)
        z_iter = iter(zs)

        # omit silence
        audio_query.pre_phoneme_length = next(phoneme_lengths_iter)
        next(z_iter)
        for mora in process_morae:
            if mora.consonant:
                mora.consonant_length = next(phoneme_lengths_iter)
                z = np.mean(ma.filled(ma.asanyarray([next(z_iter), next(z_iter)]), 0))
            else:
                z = next(z_iter)
                if ma.is_masked(z):
                    z = 0

            mora.vowel_length = next(phoneme_lengths_iter)
            mora.pitch = 5.75 + z * 0.0125

        audio_query.post_phoneme_length = next(phoneme_lengths_iter)
        pprint(audio_query.to_dict())
        try:
            resp = await audio_query.synthesis(speaker=8)

            x_, fs_ = soundfile.read(io.BytesIO(resp))
            x = librosa.resample(x, orig_sr=fs, target_sr=fs_)
            f0, _, _ = pyworld.wav2world(x, fs_, frame_period=5.)
            f0 = ma.masked_equal(f0, 0.)
            f0 = (f0 - np.mean(f0)) * 1.25 + np.mean(f0)
            f0_, sp_, ap_ = pyworld.wav2world(x_, fs_, frame_period=5.)
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
            f0 = ma.filled(f0 - np.mean(f0) + np.mean(ma.masked_equal(f0_, 0.)), np.nan)

            f0 = pd.Series(f0).interpolate(method='cubic').to_numpy(na_value=0.)

            modifier = np.copy(f0_)
            modifier[modifier > 0] = 1
            f0 = f0 * modifier

            resp = pyworld.synthesize(f0, sp_, ap_, 24000, frame_period=5.)
            soundfile.write("voice.wav", resp, samplerate=24000)
        except Exception as e:
            print(e)

        with open("voice.lab", "wb") as f:
            f.write("\n".join([
                f"{s:.7f} {e:.7f} {p}" for (s, e), (_, _, p) in zip(second_segmentation, segmentation_matches)
            ]).encode())

        # with open("reorg.wav", "wb") as f:
        #     x, fs = librosa.load("voice.wav", sr=24000, dtype=numpy.float64)
        #     _, sp, ap = pyworld.wav2world(x, fs)
        #     f.write(pyworld.synthesize(f0, sp, ap, 24000))

        await asyncio.wait_for(proc.wait(), 5)


if __name__ == '__main__':
    asyncio.run(main(), debug=True)
