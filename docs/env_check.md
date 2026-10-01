# Isaac Sim 実行環境の検証結果

## GUIでの実行検証結果

GUI上で、ExamplesからFrankaのPick and Placeを実行。
Windows タスクマネージャーから目視で計測。

| FPS | 50 ~ 60 |
| VRAM | 3.2 GiB (/ 12.0 GiB) |
| RAm | 15.9 GiB (/ 31.6 GiB) |

## headlessでの実行検証結果

headlessでfrankaのPick and Placeを実行するスクリプトを実行。

|  | カメラあり（640x480） | --no-camera |
| --- | --- | --- |
| 実時間 | 14.47 s | 11.26 s |
| シミュ時間 | 16.667 s | 16.667 s |
| 実時間比 | 1.15 | 1.48 |
| 平均FPS | 69.1 | 88.8 |
| ステップ数 | 1000 | 1000 |
| VRAMピーク | 3187 MiB（起動前 988 MiB から +2199） | 3018 MiB（+2031） |
| RAMピーク | 7172 MiB | 6934 MiB |
| 起動時間 | 8.3 s | 7.9 s |
| シーン構築 | 1.4 s | 1.3 s |
