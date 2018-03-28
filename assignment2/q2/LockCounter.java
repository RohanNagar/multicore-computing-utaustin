package q2;

public class LockCounter extends Counter {
  int count;
  MyLock lock;

  public LockCounter(MyLock lock) {
    this.lock = lock;
    this.count = 0;
  }

  @Override
  public void increment() {
    int myId;

    try {
      myId = Integer.parseInt(Thread.currentThread().getName());

      lock.lock(myId);
      count++;
      lock.unlock(myId);
    } catch (NumberFormatException e) {
      System.out.println("Thread names must be unique numbers in set [0, numThread]");
      System.exit(-1);
    }
  }

  @Override
  public int getCount() {
    return count;
  }
}

