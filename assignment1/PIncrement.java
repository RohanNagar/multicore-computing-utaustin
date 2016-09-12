import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class PIncrement {
  private static final int NUM_OPERATIONS = 1200000;
  private static final ReentrantLock lock = new ReentrantLock();

  private static int value;
  private static AtomicInteger atomicValue;

  private static int nThreads;
  private static int k;
  private static AtomicInteger[] turn;
  private static AtomicInteger[] flag;

  /**
   * Increments the variable c in parallel. There are four implementations,
   * modify the implementation used by changing the first parameter passed into the
   * runImplementation method.
   *
   * @param c The initial value of the shared variable.
   * @param numThreads The number of threads to create.
   * @return The final value of the variable c after NUM_OPERATIONS additions.
   */
  public static int parallelIncrement(int c, int numThreads) {
    // Init values needed for all types of algorithms
    value = c;
    atomicValue = new AtomicInteger(c);
    nThreads = numThreads;
    flag = new AtomicInteger[numThreads];
    turn = new AtomicInteger[numThreads];
    for(int i = 0; i < numThreads; i++)
    {
      flag[i] = new AtomicInteger(0);
      turn[i] = new AtomicInteger();
    }
    k = (int) Math.round(Math.log(numThreads)/Math.log(2));

    // Time the implementation
    long startTime = System.nanoTime();
    int result = runImplementation(Implementation.PETERSONS_TOURNAMENT, numThreads);
    long endTime = System.nanoTime();

    // Print results
    System.out.println("Time taken: " + ((endTime - startTime) / 1000000.0) + " ms");
    System.out.println("Result: \n" + result);

    return result;
  }

  /**
   * Runs the parallel increment method for a given implementation type.
   * @param implementation The type of implementation to use
   * @param numThreads The number of threads to create
   * @return The final value of the shared variable
   */
  private static int runImplementation(Implementation implementation, int numThreads) {
    Set<Thread> threads = new HashSet<>();
    int threadOperations = NUM_OPERATIONS / numThreads;

    // Create all threads and start them
    for (int i = 0; i < numThreads; i++) {
      Thread t = new ParallelIncrement(implementation, threadOperations, i);

      threads.add(t);
      t.start();
    }

    // Join all threads
    for (Thread t : threads) {
      try {
        t.join();
      } catch (InterruptedException e) {
        System.out.println("There was a concurrent error: " + e.getMessage());
      }
    }

    // Return the value
    if (implementation.equals(Implementation.ATOMIC_INTEGER)) {
      return atomicValue.get();
    } else {
      return value;
    }
  }

  /**
   * Defines all possible implementations of the parallel increment.
   */
  private enum Implementation {
    PETERSONS_TOURNAMENT,
    ATOMIC_INTEGER,
    SYNCHRONIZED,
    REENTRANT_LOCK
  }

  /**
   * This class is used as the Thread class that increments a value in parallel.
   */
  private static class ParallelIncrement extends Thread {
    private final Implementation implementation;
    private final int threadOperations;
    private final int pid;

    /**
     * Create a new instance of the ParallelIncrement class.
     * @param implementation The type of parallel increment implementation to use
     * @param threadOperations The number of times a single thread should increment
     */
    ParallelIncrement(Implementation implementation, int threadOperations, int pid) {
      this.implementation = implementation;
      this.threadOperations = threadOperations;
      this.pid = pid;
    }

    @Override
    public void run() {
      switch (implementation) {
        /** Part A */
        case PETERSONS_TOURNAMENT:
          for (int i = 0; i < threadOperations; i++) {
            PT_inc2(pid%2);
            value++;
            PT_exit();
          }

          break;

        /** Part B */
        case ATOMIC_INTEGER:
          for (int i = 0; i < threadOperations; i++) {
            int old = atomicValue.get();
            if (!atomicValue.compareAndSet(old, old + 1)) {
              i--; // redo if the compareAndSet failed
            }
          }

          break;

        /** Part C */
        case SYNCHRONIZED:
          for (int i = 0; i < threadOperations; i++) {
            addSynchronized();
          }

          break;

        /** Part D */
        case REENTRANT_LOCK:
          for (int i = 0; i < threadOperations; i++) {
            lock.lock();
            try {
              value = value + 1;
            } finally {
              lock.unlock();
            }
          }

          break;

        /** Just in case */
        default:
          System.out.println("Unknown implementation - ending thread execution.");
        }
    }

    /**
     * Locks the Peterson's Tournament Lock.
     */
    private void PT_inc2(int b) {
      int x = nThreads/2 + pid/2;
      for(int i = 1; i <= k; i++)
      {
        flag[pid].set(i);
        turn[x].set(b);
        double oppSize = Math.pow(2,i);
        int mul = pid/(int)oppSize;
        for(int j = 0; j < oppSize; j++)
        {
          while(pid != mul*2+j && turn[x].get() == b && flag[mul*2+j].get() >= i){};
        }
        b = x%2;
        x /= 2;
      }
    }

    /**
     * Unlocks the Peterson's Tournament Lock.
     */
    private void PT_exit() {
      flag[pid].set(0);
    }
  }

  /**
   * Method used in the synchronized implementation to perform the addition.
   */
  synchronized private static void addSynchronized() {
    value = value + 1;
  }
}
