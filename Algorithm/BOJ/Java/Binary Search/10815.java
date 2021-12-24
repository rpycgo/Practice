package algorithm;

//import java.util.Arrays;
//import java.util.Scanner;
import java.io.*;
import java.util.*;

public class Main {
	public static void main(String[] args) throws IOException{
		//Scanner sc = new Scanner(System.in);
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));		
		int N = Integer.parseInt(br.readLine());
		int[] holding_card_list = new int[N];
		StringTokenizer st = new StringTokenizer(br.readLine());
		
		//int N = sc.nextInt();
		for(int i = 0; i < N; i++) {			
			holding_card_list[i] = Integer.parseInt(st.nextToken());
		}
		Arrays.sort(holding_card_list);
		
		int M = Integer.parseInt(br.readLine());
		st = new StringTokenizer(br.readLine());
		Integer[] card_list_to_check = new Integer[M];
		for(int i = 0; i < M; i++) {
			card_list_to_check[i] = Integer.parseInt(st.nextToken());
		}
		
		for(int number_to_check: card_list_to_check) {
			System.out.printf("%d ", isNumberInHoldingCardList(holding_card_list, number_to_check));
		}
		
		//sc.close();
	}
	
	static int isNumberInHoldingCardList(int[] holding_card_list, int number_to_check) {
		int result = 0;
		
		int left = 0;
		int right = holding_card_list.length - 1;
		int mid;
		
		while (left <= right) {
			mid = (left + right) / 2;
			
			if (holding_card_list[mid] == number_to_check) {
				result = 1;
				break;
			}
			else if (holding_card_list[mid] > number_to_check) {
				right = mid - 1; 
			}
			else {
				left = mid + 1;
			}
		}		
		return result;
	}
}