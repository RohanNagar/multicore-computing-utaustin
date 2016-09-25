import java.util.concurrent.Semaphore;

public class CyclicBarrier {
  private static final Semaphore mutex = new Semaphore(1);
  private final int parties;
  private int count;
  private int resets;

  public CyclicBarrier(int parties) {
    this.parties = parties;
    this.count = parties;
    this.resets = 0;
  }

  synchronized int await() throws InterruptedException {
    // Acquire Semaphore so we can modify our count and reset
    mutex.acquire();
    count--;
    int index = count;

    if (index == 0) {
      // If we are at zero, all threads have arrived
      count = parties;
      resets++;
      notifyAll();

      mutex.release();
      return 0;
    } else {
      // Else, we need to wait
      int r = resets;
      while (true) {
        mutex.release();
        wait();
        mutex.acquire();

        // Resets variable will have changed if we are actually done waiting
        // Otherwise we were woken up on accident, need to keep waiting
        if (r != resets) {
          mutex.release();
          return index;
        }
      }
    }
  }

  public static void main(String[] args) {
    int threads = 5;
    CyclicBarrier cyclicBarrier = new CyclicBarrier(threads);

    for (int i = 0; i < threads; i++) {
      Thread t = new Thread() {
        @Override
        public void run() {
          System.out.println("Starting " + this.getName());
          try {
            System.out.println("Beginning to wait for all threads. " + this.getName());
            cyclicBarrier.await();
          } catch (InterruptedException e) {
            System.out.println("Interrupted.");
          }
          System.out.println("Finished waiting. Ending " + this.getName());
        }
      };

      t.start();
    }
  }
}
