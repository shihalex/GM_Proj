# 基於Yolov13應⽤於道路標誌辨識、定位與圖資⾃動化更新研究
contribute by shihalex

專案內容：
- 使用校園內與道路環境蒐集之道路標誌影像，使用 Transfer learning 與 Data argumentation 技術訓練 Yolov13 模型
- 給予一段道路影片，使用 Yolov13 進行標誌偵測，搭配 BoT-SORT 給予 id 並紀錄標誌影像與標誌 (r, c) 座標
- 使用 VGGT 根據道路影像進行建模，並使用前方交會求解標誌 GNSS 座標
- 根據 GNSS 座標，展示到地圖中，並評估其與真實標誌位置距離差異

更多專案介紹：https://www.canva.com/design/DAG95d9oV84/b1WGsgzXtRUoGgyQ_RbdpQ/edit?utm_content=DAG95d9oV84&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

專案成果報告：https://docs.google.com/document/d/10hZ0-rpSR7LnCaz_K20oTrRTM3ifEL1O/edit?usp=sharing&ouid=117538600936579205022&rtpof=true&sd=true
