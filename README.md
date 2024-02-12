# Kodama
自分の声でVoicevoxを調声したい。そんな気持ち。  
Juliusでの音素セグメンテーション + WORLDの音高解析でなんとかする。

## 機能
* Juliusによるタイミング解析による長さ補正
* Worldによるピッチ解析による音高補正
* Voicevox出力に対するピッチ強制
* vvproj形式(Voicevox プロジェクト)への出力

## INSTALL
* ``poetry install``
* [Julius](https://github.com/julius-speech/julius/releases/tag/v4.6)の実行ファイルを持ってくる
* hmmdefs(音響モデル)は[hmmdefs_monof_mix16_gid.binhmm](https://github.com/julius-speech/segmentation-kit/blob/master/models/hmmdefs_monof_mix16_gid.binhmm)でしか動作検証していない
* ``poetry run kodama -j .\julius.exe -h .\hmmdefs_monof_mix16_gid.binhmm -a .\adinrec.exe -s 8 -o output.wav -r outrec.wav -t こんにちは`` とかで動きます
* helpオプションでオプションの詳細が見れます

## 動作環境
* Windows
* Voicevoxエンジンがすでに立ち上がっていること

```
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
```