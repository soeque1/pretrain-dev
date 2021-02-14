import glob
import os
from pathlib import Path
import pytest
from tools.config import cfg_from_yaml_file, cvt_tokenizer, cvt_serialization


@pytest.fixture(scope="package")
def tokenizer_cfg():
    yaml_path = Path.cwd() / "cfgs/pipelines/word_piece_with_morpheme.yaml"
    cfg = cfg_from_yaml_file(yaml_path, cvt_tokenizer)
    cfg['Path']['data-path'] = Path.cwd() / "tests/resources/namuwiki.*.txt"
    cfg['Path']['save-path'] = Path.cwd() / "tests/resources/samples/"
    cfg['Samples']['rate'] = 0.9
    return cfg


@pytest.fixture(scope="package")
def serialization_cfg():
    yaml_path = Path.cwd() / "cfgs/serialization/pyarrow_v1.yaml"
    cfg = cfg_from_yaml_file(yaml_path, cvt_serialization)
    cfg['Path']['data-path'] = Path.cwd() / "tests/resources/namuwiki.*.txt"
    cfg['Path']['save-path'] = Path.cwd() / "tests/resources/samples/serialized"
    return cfg


@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown_package():
    test_dir = Path.cwd() / "tests/resources/"
    list_of_namuwiki_filepath = [test_dir / "namuwiki.01.txt", test_dir / "namuwiki.02.txt"]
    list_of_tests_samples = ["[목차]== 개요 ==BEMANI 시리즈의 악곡. \n",
                             "보컬은 코사카 리유, Noria. \n",
                             "롱버젼이 앨범 \"BeForU\"에 수록되었다. \n",
                             "== 팝픈뮤직 ==  * 곡 목록으로 돌아가기팝픈뮤직 9에 처음 수록되었다. \n",
                             "EX채보는 전체적으로 8비트 세로연타+동시치기 위주. \n",
                             "후반부에 밀도가 조금 높아지므로 주의. \n",
                             "17 무비 때 삭제되었다가 19 튠스트릿에서 다른 BeForU 멤버의 곡들과 함께 부활했다. \n",
                             "=== 아티스트 코멘트 ===||제 안에서 BRE∀K DOWN!에 이은 걸즈 락 노선제 2탄이 ☆shining☆인 겁니다. \n",
                             "이번에는 BeForU로부터 코사카 리유와 시리아시 노리아 두 사람을 유닛화해서,로서 전면적으로 기용했습니다. \n",
                             "shining이라는 단어는, 제가 고등학생 시절부터제 자신을 던졌던 단어로, 살아가면서잊을 리가 없는, 외상()의 마음입니다."]

    for namuwiki_filepath in list_of_namuwiki_filepath:
        with open(namuwiki_filepath, mode="w", encoding="utf-8") as io:
            for sample in list_of_tests_samples:
                io.write(sample)

    yield

    # teardown
    for namuwiki_filepath in list_of_namuwiki_filepath:
        os.remove(namuwiki_filepath)

    for i in glob.glob(str(test_dir) + '/**/*.txt', recursive=True) + glob.glob(str(test_dir) + '/**/*.parquet', recursive=True):
        os.remove(i)
