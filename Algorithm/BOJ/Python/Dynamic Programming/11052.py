def get_max_price(N: int, cards: list) -> int:
    answer = [0 for _ in range(N + 1)]

    for i in range(1, N + 1):
        for j in range(1, i + 1):
            answer[i] = max(answer[i], answer[i - j] + cards[j])

    return answer[N]


def main():
    N = int(input())
    cards = list(map(int, input().split()))
    cards.insert(0, 0)

    print(get_max_price(N, cards))


if __name__ == '__main__':
    main()
