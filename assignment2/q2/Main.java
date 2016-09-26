package q2;

import java.util.ArrayList;

public class Main {
    public static void main (String[] args) {
        Counter counter = null;
        MyLock lock;
        long executeTimeMS = 0;
        int numThread = 6;
        int numTotalInc = 1200000;

        if (args.length < 3) {
            System.err.println("Provide 3 arguments");
            System.err.println("\t(1) <algorithm>: fast/bakery/synchronized/"
                    + "reentrant");
            System.err.println("\t(2) <numThread>: the number of test thread");
            System.err.println("\t(3) <numTotalInc>: the total number of "
                    + "increment operations performed");
            System.exit(-1);
        }
        
        numThread = Integer.parseInt(args[1]);
        numTotalInc = Integer.parseInt(args[2]);
        if (args[0].equals("fast")) {
            lock = new FastMutexLock(numThread);
            counter = new LockCounter(lock);
        } else if (args[0].equals("bakery")) {
            lock = new BakeryLock(numThread);
            counter = new LockCounter(lock);
        } else {
            System.err.println("ERROR: no such algorithm implemented");
            System.exit(-1);
        }

        // TODO
        // Please create numThread threads to increment the counter
        // Each thread executes numTotalInc/numThread increments
        // Please calculate the total execute time in millisecond and store the
        // result in executeTimeMS
        int timesToInc = numTotalInc/numThread;
        ArrayList<myThread> threads = new ArrayList<myThread>(numThread);
        long startTime = System.nanoTime();
        for(int i = 0; i < numThread; i++)
        {
        	myThread t = new myThread(String.valueOf(i), timesToInc, counter);
        	threads.add(t);
        	t.start();
        }
        for(myThread t : threads){
        	try {
				t.join();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
        }
        long endTime = System.nanoTime();
        long elapsedTime = endTime - startTime;
        executeTimeMS = (long)(elapsedTime / 1000000.0);
        System.out.printf("Elapsed time: %d\n", executeTimeMS);
        System.out.printf("Counter: %d\n", counter.getCount());
    }
    
    static class myThread extends Thread{
    	int timesToInc;
    	Counter counter;
        public myThread(String name, int timesToInc, Counter counter){
        	super(name);
        	this.timesToInc = timesToInc;
        	this.counter = counter;
        }
		@Override
		public void run() {
			// TODO Auto-generated method stub
			for(int i = 0; i < timesToInc; i++)
			{
				counter.increment();
			}
		}
    
    }
}
