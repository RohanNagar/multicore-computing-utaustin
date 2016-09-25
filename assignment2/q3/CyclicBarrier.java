import java.util.concurrent.Semaphore;

public class CyclicBarrier {
  private static final Semaphore mutex = new Semaphore(1);
  private static final Semaphore barrier = new Semaphore(0);
  private static final Semaphore barrier2 = new Semaphore(1);
  private final int parties;
  private int index;
  private int count;

  public CyclicBarrier(int parties) {
    this.parties = parties;
    this.index = parties - 1;
    this.count = 0;
  }

  int await() throws InterruptedException {
    // Start by waiting for all threads to arrive
    mutex.acquire();
    int index = this.index; // Get the arrival index to return at the end
    this.index--;
    count++;
    if (count == parties) {
      // In this case, all threads have arrived. Lock barrier2 and unlock barrier.
      barrier2.acquire();
      barrier.release();
    }
    mutex.release();

    // Wait happens here on the acquire
    barrier.acquire();
    barrier.release(); // Upon resuming execution, allow other threads to continue

    // Now wait for all to resume execution, allows us to reuse the lock
    mutex.acquire();
    this.index++;
    count--;
    if (count == 0) {
      // All threads are now able to continue. Release barrier2 and lock barrier
      // for future uses of the barrier.
      barrier.acquire();
      barrier2.release();
    }
    mutex.release();

    barrier2.acquire();
    barrier2.release();

    return index;
  }


  public static void main(String[] args) {
    int threads = 5;
    CyclicBarrier cyclicBarrier = new CyclicBarrier(threads);

    for (int i = 0; i < threads; i++) {
      Thread t = new Thread() {
        @Override
        public void run() {
          int iter = 4;
          for (int i = 0; i < iter; i++) {
            System.out.println("Starting iteration " + i + " for " + this.getName());
            try {
              System.out.println("Beginning to wait for all threads. " + this.getName());
              System.out.println("Index: " + cyclicBarrier.await());
            } catch (InterruptedException e) {
              System.out.println("Interrupted.");
            }
            System.out.println("Ending iteration" + i + " for " + this.getName());
          }
        }
      };

      t.start();
    }
  }
}
