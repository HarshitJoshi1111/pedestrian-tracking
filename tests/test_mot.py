import pandas as pd

from src.mot import read_gt, read_results, read_seqinfo, split_gt, write_results

GT = """1,1,100,200,50,120,1,1,0.9
1,2,300,200,50,120,1,1,0.1
1,3,500,200,50,120,0,1,0.8
1,4,700,200,50,120,0,7,1.0
1,5,900,200,80,60,0,3,1.0
2,1,105,200,50,120,1,1,0.9
"""


def test_read_seqinfo(tmp_path):
    path = tmp_path / "seqinfo.ini"
    path.write_text("[Sequence]\nname=MOT17-09-FRCNN\nimDir=img1\nframeRate=30\nseqLength=525\n"
                    "imWidth=1920\nimHeight=1080\nimExt=.jpg\n")
    info = read_seqinfo(path)
    assert (info.frames, info.fps, info.width, info.height) == (525, 30.0, 1920, 1080)


def test_split_gt_scores_pedestrians_and_ignores_other_people(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text(GT)
    positives, ignore = split_gt(read_gt(path))
    assert positives["id"].tolist() == [1, 2, 1]
    assert ignore["id"].tolist() == [3, 4]  # unscored pedestrian and static person; the car (class 3) is neither

    positives, ignore = split_gt(read_gt(path), min_visibility=0.5)
    assert positives["id"].tolist() == [1, 1]
    assert ignore["id"].tolist() == [2, 3, 4]


def test_results_round_trip(tmp_path):
    df = pd.DataFrame({"frame": [1, 2], "id": [-1, -1], "x": [1.234, 5.0], "y": [2.0, 3.0], "w": [10.0, 11.0],
                       "h": [20.0, 21.0], "score": [0.98766, 0.01234]})
    path = tmp_path / "dets" / "MOT17-02.txt"
    write_results(df, path)
    assert path.read_text().splitlines()[0] == "1,-1,1.23,2.0,10.0,20.0,0.9877,-1,-1,-1"
    back = read_results(path)
    assert back["score"].tolist() == [0.9877, 0.0123]
    assert back["frame"].tolist() == [1, 2]
