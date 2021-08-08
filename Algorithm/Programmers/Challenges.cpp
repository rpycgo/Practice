# Hash
## 1

#include <string>
#include <vector>
#include <algorithm>

using namespace std;

string solution(vector<string> participant, vector<string> completion) 
{
    int i;
    string answer = "";

    sort(participant.begin(), participant.end());
    sort(completion.begin(), completion.end());
    
    for (i = 0; i < participant.size(); i++)
    {
        if (i <= participant.size() - 2)
        {
            if (participant[i] != completion[i])
            {
                answer = participant[i];
                break;
            }
        }
        else
        {
            answer = participant[i];
        }
    }

    return answer;
}


## 2

#include <string>
#include <vector>
#include <algorithm>

using namespace std;

bool solution(vector<string> phone_book) 
{
    int i, j;
    bool answer = true, judge = false;

    sort(phone_book.begin(), phone_book.end());
    
    for (i = 0; i < (int)phone_book.size() - 1; i++)
    {
        for (j = i + 1; j < (int)phone_book.size(); j++)
        {
            if (phone_book[j].find(phone_book[i]) == 0)
            {
                answer = false;
                judge = true;
                break;
            }
        }
        if (judge == true)
        {
            break;
        }
    }
    
    return answer;
}


## 3

#include <string>
#include <vector>
#include <map>

using namespace std;

int solution(vector<vector<string>> clothes)
{
    int i, answer = 1;
    map<string, int> vec;
    map<string, int>::iterator iter;

    for (i = 0; i < (int)clothes.size(); i++)
    {
        if (vec.count(clothes[i][1]) == 0)
        {
            vec.insert(pair<string, int>(clothes[i][1], 1));
        }
        else
        {
            vec.find(clothes[i][1])->second++;
        }
    }

    for (iter = vec.begin(); iter != vec.end(); iter++)
    {
        answer *= (iter->second + 1);
    }

    return answer - 1;
}




# Stack/Queue
## 1

vector<int> solution(vector<int> heights)
{
    int i, j, index;
    vector<int> answer(1);

    for (i = 1; i < (int)heights.size(); i++)
    {
        index = 0;
        for (j = i - 1; j >= 0; j--)
        {
            if (heights[i] < heights[j])
            {
                index = j + 1;
                break;
            }
        }
        answer.push_back(index);
    }

    return answer;
}


## 3

#include <iostream>
#include <string>
#include <vector>

using namespace std;

vector<int> solution(vector<int> progresses, vector<int> speeds)
{
    double epsilon = 1e-10;
    int i, cnt;
    vector<int> time, answer;
    vector<double> diff;

    for (i = 0; i < (int)progresses.size(); i++)
    {
        diff.push_back((double)(100 - progresses[i]) / (double)speeds[i]);
    }
        
    for (i = 0; i < (int)diff.size(); i++)
    {
        if (diff[i] - (int)diff[i] < epsilon)
        {
            time.push_back(int(diff[i]));
        }
        else
        {
            time.push_back(int(diff[i] + 1));
        }
    }
   
    for (i = 0; i < (int)time.size() - 1; i++)
    {
        if (time[i] > time[i + 1])
        {
            time[i + 1] = time[i];
        }
    }
        
    cnt = 1;
    for (i = 0; i < (int)time.size(); i++)
    {
        if (i < (int)time.size() - 1) 
        {
            if (time[i] == time[i + 1])
            {
                cnt++;
            }
            else
            {
                answer.push_back(cnt);
                cnt = 1;
            }
        }
        else
        {
            if (time[i] == time[i - 1])
            {
                answer.push_back(cnt);
            }
            else
            {
                answer.push_back(1);
            }
        }
    }

    return answer;    
}


## 6

#include <string>
#include <vector>

using namespace std;

vector<int> solution(vector<int> prices)
{
    int i, j, time;
    vector<int> answer;

    for (i = 0; i < prices.size(); i++)
    {
        time = 0;
        for (j = i + 1; j < prices.size(); j++)
        {
            time++;
            if (prices[i] > prices[j])
            {
                break;
            }
        }
        answer.push_back(time);
    }
    
    return answer;
}




# Heap
## 1

#include <string>
#include <vector>
#include <queue>

using namespace std;

int solution(vector<int> scoville, int K)
{
    int cnt = 0, i, min0, min1;
    priority_queue<int, vector<int>, greater<int>> pq;

    for (i = 0; i < (int)scoville.size(); i++) 
    {
        pq.push(scoville[i]);
    }
    
    if (pq.size() == 0)
    {
        return -1;
    }
    else if (K == 0)
    {
        return cnt;
    }
    if (pq.size() <= 1)
    {
        if (pq.top() >= K)
        {
            return cnt;
        }
        else
        {
            return -1;
        }
    }
    else
    {
        while (pq.top() < K)
        {
            cnt++;
            min0 = pq.top();
            pq.pop();
            min1 = pq.top();
            pq.pop();
            pq.push(min0 + 2 * min1);
            if (pq.size() == 1 && pq.top() < K)
            {
                return -1;
            }
        }
    }

    return cnt;
}


## 2

#include <string>
#include <vector>
#include <queue>

using namespace std;

int solution(int stock, vector<int> dates, vector<int> supplies, int k)
{
    int i, answer = 0, idx = 0;
    priority_queue<int, vector<int>, less<int>> pq;
    
    while (stock < k)
    {
        for (i = idx; i < (int)dates.size(); i++)
        {
            if (stock < dates[i])
            {
                break;
            }
            pq.push(supplies[i]);
            idx = i + 1;
        }
        
        stock += pq.top();
        pq.pop();
        answer += 1;
    }
    
    return answer;    
}




# Sort
## 1

#include <string>
#include <vector>
#include <algorithm>

using namespace std;

vector<int> solution(vector<int> array, vector<vector<int>> commands) 
{
    int i;
    vector<int> answer, temp;

    for (i = 0; i < (int)commands.size(); i++)
    {
        temp = vector<int>(array.begin() + commands[i][0] - 1, array.begin() + commands[i][1]);
        sort(temp.begin(), temp.end());
        answer.push_back(temp[commands[i][2] - 1]);
    }

    return answer;
}


## 3

#include <string>
#include <vector>

using namespace std;

int solution(vector<int> citations) 
{
    int l = citations.size(), h = 0, i;
    int upper_cit, inf_cit;

    while (true)
    {
        upper_cit = 0;
        inf_cit = 0;

        for (i = 0; i < citations.size(); i++)
        {
            if (citations[i] >= h)
            {
                upper_cit++;
            }

            if (citations[i] <= h)
            {
                inf_cit++;
            }
        }

        if (upper_cit >= h && (l - h) <= inf_cit)
        {
            break;
        }
        else
        {
            h++;
        }

    }
    
    return h;
}




# Greedy
## 1

#include <string>
#include <vector>

using namespace std;

int solution(int n, vector<int> lost, vector<int> reserve) 
{
    int i, answer = 0;
    vector<int> vec(n, 1);

    for (i = 0; i < (int)lost.size(); i++)
    {
        vec[(lost[i] - 1)] -= 1;
    }
    for (i = 0; i < (int)reserve.size(); i++)
    {
        vec[(reserve[i] - 1)] += 1;
    }

    for (i = 0; i < (int)vec.size() - 1; i++)
    {
        if (vec[i] == 0 && vec[i + 1] == 2)
        {
            vec[i] += 1;
            vec[i + 1] -= 1;
        }
    }
    for (i = (int)vec.size() - 1; 0 < i; i--)
    {
        if (vec[i] == 0 && vec[i - 1] == 2)
        {
            vec[i] += 1;
            vec[i - 1] -= 1;
        }
    }
       
    for (i = 0; i < vec.size(); i++)
    {
        if (vec[i] >= 1)
        {
            answer += 1;
        }        
    }

    return answer;
}


## 3
#include <string>
#include <vector>

using namespace std;

string solution(string number, int k) 
{
    string answer = "";
    int n = number.size() - k, start = 0;
    int _max, idx_max, i, j;
    
    for (i = 0; i < n; i++)
    {
        _max = number[start];
        idx_max = start;
        
        for(j = start; j < k + i + 1; j++)
        {
            if (_max < number[j])
            {
                _max = number[j];
                idx_max = j;
            }
        }
        
        start = idx_max + 1;
        answer += _max;
    }
    
    return answer;
}


## 4

#include <string>
#include <vector>
#include <algorithm>

using namespace std;

int solution(vector<int> people, int limit) 
{
    int start = 0, end = people.size() - 1, dup = 0;
    int answer;

    sort(people.begin(), people.end());

    while (start < end)
    {
        if (people[start] + people[end] <= limit)
        {
            dup++;
            start++;
            end--;
        }
        else
        {
            end--;
        }
    }

    answer = people.size() - dup;

    return answer;
}
