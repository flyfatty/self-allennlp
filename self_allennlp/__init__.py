# @Time : 2020/12/21 11:48
# @Author : LiuBin
# @File : __init__.py.py
# @Description : 
# @Software: PyCharm
# -*- coding: utf-8 -*-
"""
Created on 2021/3/31 17:30

@author: sun shaowen
"""
import random
import string


class generator:
    def __init__(self, length, special=False, special_chars=None):
        if length < 4:
            raise ValueError("Password length cannot be smaller than 4.")
        self.length = length
        self.special_chars = string.punctuation if not special_chars else special_chars
        self.num_location = random.randint(0, length - 1)

        if special:
            self.num_location, self.upper_location, self.lower_location, self.special_location = \
                random.sample(range(1, self.length), 4)
        else:
            self.num_location, self.upper_location, self.lower_location = \
                random.sample(range(1, self.length), 3)

    def __call__(self):
        length_ = self.length
        while length_:
            if hasattr(self, "special_location") and length_ + self.special_location == self.length:
                yield random.choice(self.special_chars)
            elif length_ + self.num_location == self.length:
                yield random.choice(string.digits)
            elif length_ + self.upper_location == self.length:
                yield random.choice(string.ascii_uppercase)
            elif length_ + self.lower_location == self.length:
                yield random.choice(string.ascii_lowercase)
            else:
                if hasattr(self, "special_location"):
                    yield random.choice(string.ascii_letters + string.digits + self.special_chars)
                else:
                    yield random.choice(string.ascii_letters + string.digits)
            length_ -= 1


def get_password(length, nums=1, **kwargs):
    cc = generator(length, **kwargs)
    return ["".join(cc()) for _ in range(nums)]


print(get_password(10, 4, special=False))
