package boj;

import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		
		int N = sc.nextInt();
		
		int[] cards = new int[N + 1];
		cards[0] = 0;
		for(int i = 1; i <= N; i++) {			
			cards[i] = sc.nextInt(); 
		}
		
		System.out.println(getMaxPrice(N, cards));
	}
	
	public static int getMaxPrice(int N, int cards[]) {
		int[] answer = new int[N + 1];
		
		for(int i = 1; i < N + 1; i++) {
			for(int j = 1; j < i + 1; j++) {
				answer[i] = getMax(answer[i], answer[i - j] + cards[j]);						
			}
		}
		
		return answer[N];
	}
	
	public static int getMax(int a, int b) {
		if (a >= b) {
			return a;
		}
		else {
			return b;				
		}		
	}
}
