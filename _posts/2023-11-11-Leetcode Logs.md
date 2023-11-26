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

### 2023-11-12

715 [Range 模块](https://leetcode.cn/problems/range-module/description/)

> 设计一个数据结构来跟踪表示为 **半开区间** 的范围并查询它们。
>
> **半开区间** `[left, right)`
>
> 实现增删查

**平衡二叉搜索树的模板题**

在 C++ STL 中为 `set`

树上每个节点代表一个区间，以区间的左端点 left 作为 key，按照 left 从小到大排序

注意到：由于会引入区间合并操作，所以树上的节点不会有重叠

在代码过程中有两个细节需要注意：

+ `ranges.lower_bound`：返回第一个 key **大于等于** left 的节点，这里要额外处理**等于**的情况

+ 需要处理左端点小于 left，但是右端点大于 left 的重叠情况，所以上一步返回的迭代器需要向前看一个：

  ```c++
  if(it != ranges.begin() && (--it)->second < left) it++;
  ```

另外学到了：

```c++
it = ranges.erase(it);
```

这样去迭代。

以及引用 vector 中的元素：

```c++
it = ranges.erase(it);
```

完整代码：

```c++
class RangeModule {
public:
    RangeModule() {}

    void addRange(int left, int right) {
        auto it = ranges.lower_bound({ left, right }); 
        if (it != ranges.begin() && (--it)->second < left) it++;
        while (it != ranges.end() && it->first <= right) {
            left = min(left, it->first);
            right = max(right, it->second);
            it = ranges.erase(it);
        }
        ranges.insert({ left, right });
    }

    bool queryRange(int left, int right) {
        auto it = ranges.lower_bound({ left, right });
        if (it->first == left) {
            return it->second >= right;
        }
        else {
            if (it == ranges.begin()) return false;
            it--;
            return it->second >= right;
        }
    }

    void removeRange(int left, int right) {
        auto it = ranges.lower_bound({ left, right });
        if (it != ranges.begin() && (--it)->second < left) it++;
        vector<pair<int, int>> tmp;
        while (it != ranges.end() && it->first < right) {
            if (it->first < left) tmp.push_back({ it->first, left });
            if (it->second > right) tmp.push_back({ right, it->second });
            it = ranges.erase(it);
        }
        for (const auto& t : tmp) ranges.insert(t);
    }

private:
    set<pair<int, int>> ranges;
};
```

### 2023-11-13

307 [区域和检索 - 数组可修改 ](https://leetcode.cn/problems/range-sum-query-mutable/description/)

> 给你一个数组 `nums` ，请你完成两类查询。
>
> 1. 其中一类查询要求 **更新** 数组 `nums` 下标对应的值
> 2. 另一类查询要求返回数组 `nums` 中索引 `left` 和索引 `right` 之间（ **包含** ）的nums元素的 **和** ，其中 `left <= right`

**基础版线段树的模板题**

只涉及到加法，而且修改是单点修改，所以比较简单的模板。~~虽然好久没写已经忘了~~

值得关注的是求区间和，有一些细节需要注意。

完整代码：

```c++
class NumArray {
public:
    NumArray(vector<int>& nums) {
        if(nums.empty()) return;
        n = nums.size();
        tree.resize(2 * n);
        buildTree(nums);
    }
    
    void update(int index, int val) {
        index += n;
        tree[index] = val;
        while(index > 0){
            int left = index;
            int right = index;
            if(index % 2 == 0){
                right = index + 1;
            }
            else{
                left = index - 1;
            }
            tree[index / 2] = tree[left] + tree[right];
            index /= 2;
        }
    }
    
    int sumRange(int left, int right) {
        left += n;
        right += n;
        int sum = 0;
        while(left <= right){
            if(left % 2 == 1){
                sum += tree[left];
                left++;
            }
            if(right % 2 == 0){
                sum += tree[right];
                right--;
            }
            left /= 2;
            right /= 2;
        }
        return sum;
    }

private:
    vector<int> tree;
    int n;

    void buildTree(vector<int>& nums){
        for(int i = n; i < 2 * n; i++){
            tree[i] = nums[i - n];
        }
        for(int i = n - 1; i > 0; i--){
            tree[i] = tree[i * 2] + tree[i * 2 + 1];
        }
    }
};
```

### 2023-11-14

1334 [阈值距离内邻居最少的城市](https://leetcode.cn/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/description/)

> 有 `n` 个城市，按从 `0` 到 `n-1` 编号。给你一个边数组 `edges`，其中 `edges[i] = [fromi, toi, weighti]` 代表 `fromi` 和 `toi` 两个城市之间的双向加权边，距离阈值是一个整数 `distanceThreshold`。
>
> 返回能通过某些路径到达其他城市数目最少、且路径距离 **最大** 为 `distanceThreshold` 的城市。如果有多个这样的城市，则返回编号最大的城市。
>
> n <= 100

**Dijkstra 模板题**

只需要以每个城市为起点，Dijkstra，记录到所有城市的最短距离，统计有多少在 `distanceThreshold` 以内，最后比较。

可以剪枝：在 Dijkstra 入队时，如果已经超过了 `distanceThreshold`，那么不入队。

时间复杂度：$O(n^2log(n))$

具体代码细节上，学到了：

+ vector 的 `push_bach` 和 `emplace_back` 的不同，前者是将已经创建好的对象直接插入，后者是提供参数，在 vector 内创建。后者效率要高一些。

+ vector 初始化：

  ```c++
  vector<int> dist(graph.size(), INT_MAX);
  ```

完整代码：

```c++
class Solution {
public:
    int findTheCity(int n, vector<vector<int>>& edges, int distanceThreshold) {
        graph.resize(n);
        for(auto& edge: edges){
            graph[edge[0]].emplace_back(edge[1], edge[2]);
            graph[edge[1]].emplace_back(edge[0], edge[2]);
        }
        for(int i = 0; i < n; i++){
            vector<int> dist = dijkstra(i, distanceThreshold);
            int count = 0;
            for(int d : dist){
                if(d <= distanceThreshold){
                    count++;
                }
            }
            if(count <= minCount){
                minCount = count;
                minCity = i;
            }
        }
        return minCity;
    }

    vector<int> dijkstra(int start, int threshold){
        vector<int> dist(graph.size(), INT_MAX);
        dist[start] = 0;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        pq.emplace(0, start);

        while(!pq.empty()){
            auto [d, u] = pq.top();
            pq.pop();
            if(d > dist[u]) continue;
            for(auto& [v, w] : graph[u]){
                if(dist[v] > dist[u] + w){
                    dist[v] = dist[u] + w;
                    if(dist[v] <= threshold){
                        pq.emplace(dist[v], v);
                    }
                }
            }
        }
        return dist;
    }
    
private:
    vector<vector<pair<int, int>>> graph; // graph[i] = {j, w} 表示 i 到 j 的距离为 w
    int minCount = INT_MAX;
    int minCity = -1;
};
```

### 2023-11-15

2656 [K 个元素的最大和](https://leetcode.cn/problems/maximum-sum-with-exactly-k-elements/description/)

比较简单，不做记录。

### 2023-11-16

2760 [最长奇偶子数组](https://leetcode.cn/problems/longest-even-odd-subarray-with-threshold/description/)

也比较简单，假如数据范围大点必须要用 DP，那么还有记录的必要。

### 2023-11-17

2736 [最大和查询](https://leetcode.cn/problems/maximum-sum-queries/description/)

> 给你两个长度为 `n` 、下标从 **0** 开始的整数数组 `nums1` 和 `nums2` ，另给你一个下标从 **1** 开始的二维数组 `queries` ，其中 `queries[i] = [xi, yi]` 。
>
> 对于第 `i` 个查询，在所有满足 `nums1[j] >= xi` 且 `nums2[j] >= yi` 的下标 `j` `(0 <= j < n)` 中，找出 `nums1[j] + nums2[j]` 的 **最大值** ，如果不存在满足条件的 `j` 则返回 **-1** 。
>
> 返回数组 `answer` *，*其中 `answer[i]` 是第 `i` 个查询的答案
>
> - `nums1.length == nums2.length` 
> - `n == nums1.length `
> - `1 <= n <= 10^5`
> - `1 <= queries.length <= 10^5`

首先，根据数据范围，直接暴力两重循环会超时。

最早考虑将 nums1 排序，对每个查询，找到符合条件的下标（二分查找），然后只遍历符合条件的下标，但是时间复杂度并没有下降很多，极端情况下，仍然需要遍历所有 nums1。所以仍然会超时。

~~所以只能看题解了~~

首先一个关键点是：

> 找出 `nums1[j] + nums2[j]` 的 **最大值** 

即，每次求和时，nums1 和 nums1 的下标需要一致。

两个技巧：

+ 将 nums1 和 nums2 合并到一个 pair 数组，保证了求和时下标一致，然后将 nums 以及查询数组**按照 x 降序排序**

  这样在访问到某个查询时，能方便地保留所有 x 满足条件的 nums

+ 维护一个 nums 的**单调栈**，按照 **x + y** 从底到顶从大到小

  这样对于某个查询，单调栈里的 nums 的 x 一定是满足的；那就二分查找第一个满足 y 条件的即为此次查询的 ans

  那么每次访问到 nums，如果 x + y 比栈顶大，那么弹栈，如果当前 nums 的 y 比栈顶的大，那么入栈。

  **关键：如果当前访问到的 y 比之前的都要小，那么 x + y 一定小于栈中元素。所以能入栈的，一定是 y 比较大的。所以栈中元素不仅 x + y 从大到小，x 从大到小，而且 y 是从小到大。**

概括来说：

+ nums 按照 x 从大到小排序，单调栈从底到顶按 x + y 从大到小排序，那么访问 nums 的顺序保证了：栈中 x 从大到小，y 从小到大。
+ **直观理解是，保留了要么 x 有优势，要么 y 有优势的 nums**

核心总结起来，就是一句话：

**如果一个 num 的 x 和 y 都比另一个的大，那么另一个一定不会是 ans**

完整代码：

```c++
class Solution {
public:
    vector<int> maximumSumQueries(vector<int>& nums1, vector<int>& nums2, vector<vector<int>>& queries) {
        int n = nums1.size();
        int m = queries.size();
        vector<pair<int, int>> sortedNums;
        vector<tuple<int, int, int>> sortedQueries;
        for (int i = 0; i < n; i++) {
            sortedNums.emplace_back(nums1[i], nums2[i]);
        }
        sort(sortedNums.begin(), sortedNums.end(), greater<pair<int, int>>());
        for (int i = 0; i < m; i++) {
            sortedQueries.emplace_back(queries[i][0], queries[i][1], i);
        }
        sort(sortedQueries.begin(), sortedQueries.end(), greater<tuple<int, int, int>>());
        vector<pair<int, int>> stack;
        vector<int> answer(m, -1);
        int j = 0;
        for (auto [x, y, i] : sortedQueries) {
            while (j < n && sortedNums[j].first >= x) {
                auto [num1, num2] = sortedNums[j];
                while (!stack.empty() && stack.back().second <= num1 + num2) {
                    stack.pop_back();
                }
                if (stack.empty() || stack.back().first < num2) {
                    stack.emplace_back(num2, num1 + num2);
                }
                j++;
            }
            int k = binary_search(stack, y);
            if (k < stack.size()) {
                answer[i] = stack[k].second;
            }
        }
        return answer;
    }

    int binary_search(vector<pair<int, int>>& stack, int y) {
        int l = 0, r = stack.size();
        while (l < r) {
            int mid = (l + r) / 2;
            if (stack[mid].first >= y) {
                r = mid;
            }
            else {
                l = mid + 1;
            }
        }
        return l;
    }
};
```

值得关注的是，二分查找，`r = stack.size()`，而不是 `stack.size() - 1`，以处理边界情况。

### 2023-11-18

2342 [数位和相等数对的最大和](https://leetcode.cn/problems/max-sum-of-a-pair-with-equal-sum-of-digits/description/)

> 给你一个下标从 **0** 开始的数组 `nums` ，数组中的元素都是**正**整数。请你选出两个下标 `i` 和 `j`（`i != j`），且 `nums[i]` 的数位和 与 `nums[j]` 的数位和相等。
>
> 请你找出所有满足条件的下标 `i` 和 `j` ，找出并返回 `nums[i] + nums[j]` 可以得到的**最大值**。
>
> 数位和，即各位的和

比较简单，可以使用 `unordered_map` 实现：

```c++
unordered_map<int, pair<int, int>> digitSums;
```

维护一个数位和对应的最大值和次最大值，值得注意的一个细节是不存在满足条件的数对，返回 -1 

### 2023-11-19

689 [三个无重叠子数组的最大和](https://leetcode.cn/problems/maximum-sum-of-3-non-overlapping-subarrays/description/)

> 给你一个整数数组 `nums` 和一个整数 `k` ，找出三个长度为 `k` 、互不重叠、且全部数字和（`3 * k` 项）最大的子数组，并返回这三个子数组。
>
> 以下标的数组形式返回结果，数组中的每一项分别指示每个子数组的起始位置（下标从 **0** 开始）。如果有多个结果，返回字典序最小的一个。

一开始想用 DP（其实是看到题目标签里有），但是想了想，觉得状态转移方程不是很好写

**滑动窗口**

维护三个滑动窗口，长度均为 k，并且同步向右移。win1, win2, win3

维护三个当前最大值，分别是一个子数组、两个子数组、三个子数组。max1, max2, max3

**关键：每次移动，win1 比较然后更新 max1，max1 + win2 比较然后更新 max2，max2 + win3 比较然后更新 max3**

记录好下标，这里别想当然，每个 max 的多个下标需要分别记录

完整代码：

```c++
class Solution {
public:
    vector<int> maxSumOfThreeSubarrays(vector<int>& nums, int k) {
        int win1 = 0, win2 = 0, win3 = 0;
        int max1 = 0, max2 = 0, max3 = 0;
        int index1 = 0;
        int index2_1 = 0, index2_2 = 0;
        int index3_1 = 0, index3_2 = 0, index3_3 = 0;
        int n = nums.size();
        for(int i = 2 * k; i < n; i++){
            win1 += nums[i - 2 * k]; win2 += nums[i - k]; win3 += nums[i];
            if(i < 3 * k - 1) continue;
            if(win1 > max1){
                max1 = win1;
                index1 = i - 3 * k + 1;
            }
            if(max1 + win2 > max2){
                max2 = max1 + win2;
                index2_1 = index1; index2_2 = i - 2 * k + 1;
            }
            if(max2 + win3 > max3){
                max3 = max2 + win3;
                index3_1 = index2_1; index3_2 = index2_2; index3_3 = i - k + 1;
            }
            win1 -= nums[i - 3 * k + 1]; win2 -= nums[i - 2 * k + 1]; win3 -= nums[i - k + 1];
        }
        vector<int> ans;
        ans.push_back(index3_1); ans.push_back(index3_2); ans.push_back(index3_3);
        return ans;
    }
};
```

### 2023-11-20

53 [最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/)

比较简单的 DP，不做记录

### 2023-11-21

2261 [美化数组的最少删除数](https://leetcode.cn/problems/minimum-deletions-to-make-array-beautiful/description/)

模拟题，~~或者说我没想到其他有技巧的解法~~，所以不做记录

### 2023-11-22

2304 [网格中的最小路径代价](https://leetcode.cn/problems/minimum-path-cost-in-a-grid/description/)

简单的 DP，需要注意的是题目中关于数组的描述：

> `moveCost[i][j]` 是从值为 `i` 的单元格移动到下一行第 `j` 列单元格的代价

不是位置，而是**值**

其他的不需要记录

### 2023-11-23

1410 [HTML 实体解析器](https://leetcode.cn/problems/html-entity-parser/submissions/484239493/)

字符串替换题目，比较经典，所以记录一下。代码值得记住。

另外有两个小细节：

+ `&amp` 会替换为 `&`，`&` 又是 HTML 特殊字符的开头，所以应该最后替换
+ 即使最后替换 `&amp`，每次替换也应该从上次替换的下一个开始，要不然也会有被替换出的 `&` 与后续字符拼接而被误替换的情况

完整代码：

```c++
string entityParser(string text) {
    vector<string> str = {"&quot;", "&apos;", "&gt;", "&lt;", "&frasl;", "&amp;"};
    vector<string> rep = {"\"", "\'", ">", "<", "/", "&"};
    for(int i = 0; i < str.size(); i++){
        int pos = text.find(str[i]);
        while(pos != string::npos){
            text.replace(pos, str[i].size(), rep[i]);
            pos = text.find(str[i], pos + rep[i].size());
        }
    }
    return text;
}
```

### 2023-11-24

今天的每日一题 2824 [统计和小于目标的下标对数目](https://leetcode.cn/problems/count-pairs-whose-sum-is-less-than-target/description/) 比较简单，不做记录。

不过今天做的另外一道题 2750 [将数组划分成若干好子数组的方式 ](https://leetcode.cn/problems/ways-to-split-array-into-good-subarrays/) 值得记录一下：

> 数组 nums，只包含 0 和 1 两种元素
>
> 切分为若干子数组，每个子数组有且只有一个 1，问切分方法数

首先，子数组应该是不改变原数组元素的位置的

关键的想法：

以数组 `[0,1,0,0,1]` 为例，**应该是在两个 1 之间切分**，共 3 个位置，那么就有 3 种方法数，当有更多 1 时，结果相乘。

这个想明白以后，代码就比较简单了。

### 2023-11-25

今天的每日一题 1457 [二叉树中的伪回文路径](https://leetcode.cn/problems/pseudo-palindromic-paths-in-a-binary-tree/description/) 比较简答的 DFS，不做记录。

今天做的另一道题，1383 [最大的团队表现值](https://leetcode.cn/problems/maximum-performance-of-a-team/description/) 值得记录一下：

> 给定两个整数 `n` 和 `k`，以及两个长度为 `n` 的整数数组 `speed` 和` efficiency`。现有 `n` 名工程师，编号从 `1` 到 `n`。其中 `speed[i]` 和 `efficiency[i]` 分别代表第 `i` 位工程师的速度和效率。
>
> 从这 `n` 名工程师中最多选择 `k` 名不同的工程师，使其组成的团队具有最大的团队表现值。
>
> **团队表现值** 的定义为：一个团队中「所有工程师速度的和」乘以他们「效率值中的最小值」。
>
> 请你返回该团队的最大团队表现值，由于答案可能很大，请你返回结果对 `10^9 + 7` 取余后的结果。

这个题与 11 月 17 日做的 2736 最大和查询 有些相似，但是比那个简单。

核心思想就是将工程师按照效率从大到小排序，逐个考虑效率，维护一个当前选择的效率最小值。维护一个优先队列，存储效率大于当前维护的最小效率的工程师的速度。优先队列容量最大为 k，并且维护一个当前单调栈中速度的和。

两个点需要注意：

+ 一个是题目描述中的「最多选择 `k` 名不同的工程师」，也就是说可以选择少于 k 人
+ 另一个是，答案虽然需要取 MOD，但是在中途比较的过程中不可以取 MOD

代码比较简单，不做记录。

### 2023-11-26

828 [统计子串中的唯一字符 ](https://leetcode.cn/problems/count-unique-characters-of-all-substrings-of-a-given-string/description/)

> 给一个字符串，仅由大写字母组成
>
> 求该字符串的「所有字串的唯一字符」的个数之和
>
> `1 <= s.length <= 105`

一开始一直在 DP 的方向思考（~~主要是题目标签里有动态规划~~），想的是类似线段树的思想构造一个数，维护长度为 2 的幂的区间的各个字符出现的次数。

问题是创建这棵树时间为 $O(nlogn)$，但是查询仍然是 $O(n^2logn)$，而且常数比较大，至少为 26（大写字母的个数），会超时。

~~然后就去看题解了~~

核心在于转变思考切入点，之前一直在思考怎么逐个区间求，但是转变到从每个字母入手，求其对答案的贡献，那就很简单了。

对于一个字符 c，当前下标为 index，其左边最近的 c 的位置为 l，右边最近的 c 的位置为 r，那么该字符 c 对答案的贡献为 **(index - l) * (r - index)**

维护每个字母出现的所有位置，用一个二维数组即可，注意**处理好边界**。

完整代码：

```c++
class Solution {
public:
    int uniqueLetterString(string s) {
        int n = s.size();
        vector<vector<int>> f(26);
        for (int i = 0; i < n; i++) {
            f[s[i] - 'A'].push_back(i);
        }
        int ans = 0;
        for(int i = 0; i < 26; i++){
            int len = f[i].size();
            for(int j = 0; j < len; j++){
                int l = (j == 0 ? -1 : f[i][j - 1]);
                int r = (j == len - 1 ? n : f[i][j + 1]);
                ans += (f[i][j] - l) * (r - f[i][j]);
            }
        }
        return ans;
    }
};
```

