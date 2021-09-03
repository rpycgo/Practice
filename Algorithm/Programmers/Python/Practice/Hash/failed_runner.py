def solution(participant, completion):
    temp = 0
    dictionary = {}

    for part in participant:
        dictionary[hash(part)] = part
        temp += int(hash(part))

    for com in completion:
        temp -= hash(com)

    return dictionary[temp]
