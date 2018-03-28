package q2;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;

public class FastMutexLock implements MyLock {
  AtomicIntegerArray flags; // {0 = down, 1 = up}
  volatile AtomicInteger x;
  volatile AtomicInteger y;

  public FastMutexLock(int numThread) {
    flags = new AtomicIntegerArray(numThread);
    x = new AtomicInteger(-1);
    y = new AtomicInteger(-1);
  }

  @Override
  public void lock(int myId) {
    while (true) {
      flags.set(myId, 1);
      x.set(myId);

      if (y.get() != -1) {
        flags.set(myId, 0);
        while (y.get() != -1) {};
        continue;
      }

      y.set(myId);

      if (x.get() == myId) {
        return;
      } else {
        flags.set(myId, 0);

        for (int i = 0; i < flags.length(); i++) {
          while (flags.get(i) != 0) {};
        }

        if (y.get() == myId) {
          return;
        } else {
          while (y.get() != -1) {};
          continue;
        }
      }
    }
  }

  @Override
  public void unlock(int myId) {
    y.set(-1);
    flags.set(myId, 0);
  }
}

