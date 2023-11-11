---
title: LeetCode
author: xuao
date: 2023-11-11 16:15:00 +800
categories: [LeetCode]
tags: [LeetCode, C++, C, Algorithm]
---

# 解题记录

### 2023-11-11

765 [情侣牵手](https://leetcode.cn/problems/couples-holding-hands/description/)

> `n` 对情侣坐在连续排列的 `2n` 个座位上，想要牵到对方的手。
>
> 人和座位由一个整数数组 `row` 表示，其中 `row[i]` 是坐在第 `i `个座位上的人的 **ID**。情侣们按顺序编号，第一对是 `(0, 1)`，第二对是 `(2, 3)`，以此类推，最后一对是 `(2n-2, 2n-1)`。
>
> 返回 *最少交换座位的次数，以便每对情侣可以并肩坐在一起*。 *每次*交换可选择任意两人，让他们站起来交换座位。

简单的**模拟题**

首先有两个观察：对于最终的理想位次，满足：

+ 偶数位置上的人，其下一个位置为其情侣
+ ID 为偶数的人，其情侣 ID 为其 ID + 1；反之，ID 为奇数的人，其情侣 ID 为其 ID - 1

那么，只需要从前往后遍历所有偶数位次，检查其下一个位置是否为其情侣，不是的话就将其情侣交换过来。显然，这样也是交换次数最少的方案。

代码：

```c++
class Solution {
public:
    int minSwapsCouples(vector<int>& row) {
        int n = row.size();
        int ans = 0;
        vector<int> index(n);
        for (int i = 0; i < n; i++) {
            index[row[i]] = i;
        }
        for (int i = 0; i < n; i += 2) { // 遍历每个偶数位置
            int pos = find(row[i], index, row);
            if (pos == -1) continue;
            else {
                int temp = row[i + 1];
                row[i + 1] = row[pos];
                row[pos] = temp;
                index[row[i + 1]] = i + 1;
                index[row[pos]] = pos;
                ans++;
            }
        }
        return ans;
    }
    int find(int x, vector<int>& index, vector<int>& row)
    {
        // 位置在偶数位，那么检测其下一位
        if (x % 2 == 0) { // ID 为偶数，其情侣 ID 应为 x + 1  
            if (row[index[x] + 1] == x + 1) return -1;
            else return index[x + 1];
        }
        else {
            if (row[index[x] + 1] == x - 1) return -1;
            else return index[x - 1];
        }
    }
};
```

