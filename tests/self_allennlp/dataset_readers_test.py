# @Time : 2020/12/18 16:45
# @Author : LiuBin
# @File : dataset_readers_test.py.py
# @Description : 
# @Software: PyCharm
import os
import pytest

from allennlp.common.util import ensure_list

from self_allennlp import ClsTsvDataSetReader
from tests import TSET_DATA_PATH


class TestClsTsvDataSetReader:
    test_data_path = os.path.join(TSET_DATA_PATH, "cls_tsv_test.tsv")

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = ClsTsvDataSetReader(lazy=lazy)
        instances = reader.read(self.test_data_path)
        instances = ensure_list(instances)

        instance1 = {"tokens": ["The", "actors", "are", "fantastic", "."], "label": "4"}
        instance2 = {"tokens": ["It", "was", "terrible", "."], "label": "0"}
        instance3 = {"tokens": ["Chomp", "chomp", "!"], "label": "2"}

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance1["tokens"]
        assert fields["label"].label == instance1["label"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == instance2["tokens"]
        assert fields["label"].label == instance2["label"]
        fields = instances[2].fields
        assert [t.text for t in fields["tokens"].tokens] == instance3["tokens"]
        assert fields["label"].label == instance3["label"]
