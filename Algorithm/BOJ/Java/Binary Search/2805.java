package BOJ;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int M = sc.nextInt();
		int i, tree;
		
		ArrayList<Integer> trees = new ArrayList<Integer>();		
		for(i = 0; i < N; i++) {
			tree = sc.nextInt();
			trees.add(tree);
		}
		Collections.sort(trees);
		
		System.out.println(findMaxHeight(trees, M));
	}
	
	static long findMaxHeight(ArrayList<Integer> trees, int target_log) {
		int start = 1;
		int end = Collections.max(trees);
		int mid;
		
		while (start <= end) {
			mid = (start + end) / 2;
			long log = 0;
			
			for(int tree: trees) {
				if (tree > mid) {
					log += tree - mid;
				}
			}
			
			if (log >= target_log) {
				start = mid + 1;
			} else {
				end = mid - 1;
			}
		}
		
		return end;
	}
}
