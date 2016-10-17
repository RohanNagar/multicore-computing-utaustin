package q5;

public class QueueTest {
	MyQueue queue;
	int numEnqueue;
	int numDequeue;
	static final int NUM_QUEUES = 2;
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int numEnq = 10;
		int numDeq = 5;
		QueueTest lockQTester = new QueueTest(numEnq, numDeq);
		lockQTester.run();
	}
	
	public QueueTest(int numEnqueue, int numDequeue)
	{
		this.numEnqueue = numEnqueue;
		this.numDequeue = numDequeue;
	}
	
	public void run()
	{
		String queueType;
		Thread enq;
		Thread deq;
		for(int i = 0; i < NUM_QUEUES; i++)
		{
			switch(i)
			{
				default:
				case 0:
					queue = new LockQueue();
					queueType = "Lock Queue";
					break;
				case 1:
					queue = new LockFreeQueue();
					queueType = "Lock Free Queue";
					break;
			}
			
			enq = new Thread(new Enqueuer(numEnqueue));
			deq = new Thread(new Dequeuer(numDequeue));
			enq.start();
			deq.start();
			try
			{
				enq.join();
				deq.join();
			}catch (Exception e)
			{
				e.printStackTrace(System.out);
				System.exit(-1);
			}
		
			System.out.println(queue);
			boolean passed = true;
			for(int j = numDequeue; j < numEnqueue; j++)
			{
				if(!(queue.contains(j)))
				{
					System.out.println("error: doesn't contain " + j);
					passed = false;
					break;
				}
			}
			
			if(passed)
			{
				System.out.println(queueType + " tests pass");
			}
		}
		
		
	}
	
	private class Enqueuer implements Runnable{
		int numItems;
		public Enqueuer(int numItems)
		{
			this.numItems = numItems;
		}
		@Override
		public void run() {
			// TODO Auto-generated method stub
			for(int i = 0; i < numItems; i++)
			{
				
				System.out.println("Enq: " + i + " " + queue.enq(i));
			}
		}
		
	}
	
	private class Dequeuer implements Runnable{
		int numItems;
		public Dequeuer(int numItems)
		{
			this.numItems = numItems;
		}
		@Override
		public void run() {
			// TODO Auto-generated method stub
			Integer dequeued;
			for(int i = 0; i < numItems; i++)
			{
				while((dequeued = queue.deq()) == null);
				System.out.println("deq: " + dequeued);
			}
		}
		
	}

}
