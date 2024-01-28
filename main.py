import asyncio
from asyncio import StreamWriter, StreamReader
from collections.abc import Collection, Iterable

from voicevox import Client

from yomi2voca import yomi2voca


class JuliusClient:
    def __init__(self, reader: StreamReader, writer: StreamWriter):
        self._reader = reader
        self._writer = writer

    async def receive_julius_output(self) -> str:
        r = await self._reader.readuntil(b'.\n')
        return r[:-3].decode('shift_jis')  # remove dot and \n

    async def change_gramer(self, voca):
        words = self.make_words(voca)

        gram_dfa = self.make_sequential_gram_dfa(len(words))
        gram_dict = self.make_gram_dict(words)

        print(gram_dict)
        print(gram_dfa)

        self._writer.writelines([
            b'CHANGEGRAM\n',
            gram_dfa.encode('utf-8'),
            b'\nDFAEND\n',
            gram_dict.encode('utf-8'),
            b'\nDICEND\n'
        ])

        await self._writer.drain()
        return await self.receive_julius_output()

    @staticmethod
    def make_words(voca: str, silence_at_ends: bool = True) -> list[str]:
        words = [
            'silB',
            voca,
            'silE'
        ] if silence_at_ends else [voca]

        return words

    @staticmethod
    def make_gram_dict(words: Iterable[str]) -> str:
        result = []
        for i, word in enumerate(words):
            result.append(f"{i} [w_{i}] {word}")

        return '\n'.join(result)

    @staticmethod
    def make_sequential_gram_dfa(count: int) -> str:
        return '\n'.join(
            [f"{i} {count - i - 1} {i + 1} 0 {1 if i == 0 else 1}" for i in range(count)]
            + [f"{count} -1 -1 1 0"]
        )


async def receive_julius_output(reader: StreamReader) -> str:
    r = await reader.readuntil(b'.\n')
    return r[:-3].decode('utf-8')  # remove dot and \n


async def main():
    reader, writer = await asyncio.open_connection("127.0.0.1", "10500")
    julius = JuliusClient(reader, writer)
    # writer.write(b'STATUS\n')
    # r = await receive_julius_output(reader)
    # print(r)
    # writer.write(b'CURRENTPROCESS\n_default\n')
    # r = '\n'.join([
    #     await receive_julius_output(reader),
    #     await receive_julius_output(reader)
    # ])
    # print(r)

    async with Client() as client:
        audio_query = await client.create_audio_query(
            "きょーわ。いいてんきです", speaker=8
        )

        morae = [mora
                 for accent_phrase in audio_query.accent_phrases
                 for mora in accent_phrase.moras + ([accent_phrase.pause_mora] if accent_phrase.pause_mora else [])
                 ]

        print([yomi2voca(mora.text) for mora in morae])

        voca = ' '.join([yomi2voca(mora.text) for mora in morae])

        r = await julius.change_gramer(voca)
        print("> changegram")
        print(r)

        writer.write(b'SYNCGRAM\n')
        await writer.drain()
        r = await julius.receive_julius_output()
        print("> syncgram")
        print(r)

        writer.write(b'GRAMINFO\n')
        await writer.drain()
        print("> resume")
        r = await julius.receive_julius_output()
        print(r)

        writer.write(b'ACTIVATEPROCESS\n_default\n')
        print("> activateprocess")
        await writer.drain()
        r = await julius.receive_julius_output()
        print(r)
        writer.write(b'RESUME\n')
        await writer.drain()
        print("> resume")
        while True:
            r = await julius.receive_julius_output()
            print(r)


if __name__ == '__main__':
    asyncio.run(main(), debug=True)
