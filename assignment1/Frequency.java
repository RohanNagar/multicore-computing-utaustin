package com.utaustin.multicore;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Frequency {
  public static int parallelFreq(int x, int[] A, int numThreads) {
    ExecutorService threadPool = Executors.newCachedThreadPool();
    Set<Future<Integer>> futures = new HashSet<>();
    int subarraySize = A.length / numThreads;
    int startIndex = 0;

    // Submit tasks
    for (int i = 0; i < numThreads; i++) {
      Callable<Integer> callable
          = new FreqComputation(x, Arrays.copyOfRange(A, startIndex, startIndex + subarraySize));
      Future<Integer> future = threadPool.submit(callable);
      futures.add(future);

      startIndex += subarraySize;
    }

    // Calculate sum
    int frequency = 0;
    for (Future<Integer> f : futures) {
      try {
        frequency += f.get();
      } catch (InterruptedException | ExecutionException e) {
        System.out.println("A concurrent exception occurred: " + e.getMessage());
      }
    }

    threadPool.shutdown();
    return frequency;
  }

  private static class FreqComputation implements Callable {
    private final int x;
    private final int[] A;

    FreqComputation(int x, int[] A) {
      this.x = x;
      this.A = A;
    }

    public Integer call() {
      int count = 0;
      for (Integer i: A) {
        if (i == x) count++;
      }

      return count;
    }
  }

}
