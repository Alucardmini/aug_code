# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/8/19 10:31 AM'


def manacher(s:str) -> int:
    s = '#' + '#'.join(s) + '#'
    radius = [0 for _ in range(len(s))]
    maxlen, pos, max_right = 0, 0, 0

    for i in range(len(s)):
        if i < max_right:
            radius[i] = min(max_right - i, radius[pos * 2 - i])
        else:
            radius[i] = 1
        while i - radius[i] >= 0 and i + radius[i] < len(s) and s[i - radius[i]] == s[i + radius[i]]:
            radius[i] += 1
        if i - 1 + radius[i] > max_right:
            max_right = i - 1 + radius[i]
            pos = i
        maxlen = max(maxlen, radius[i])
    return maxlen - 1


def get_next_list(s: str)->list:
    next_list = [0 for _ in range(len(s))]
    for i in range(len(s)):
        if i == 0:
            next_list[i] = 0
        else:
            cnt = 0
            for k in range(0, i):
                if s[0: k] == s[i - k: i]:
                    cnt += 1
                    next_list[i] = max(next_list[i], cnt)
            # if next_list[i] == 0:
            #     next_list[i] = 1
    return next_list


# 2. kmp匹配算法
def kmp_index_sub_list(src_str, target_str):
    i, j = 0, 0
    target_next = get_next_list(target_str)
    while i < len(src_str) and j < len(target_str):
        if src_str[i] == target_str[j]:
            i += 1
            j += 1
        elif j != 0:
            j = target_next[j - 1]
        else:
            i += 1

    if j >= len(target_str):
        return i - len(target_str)
    else:
        return -1

if __name__ == '__main__':
    print(manacher('caba'))

    print(get_next_list("ABABAA"))