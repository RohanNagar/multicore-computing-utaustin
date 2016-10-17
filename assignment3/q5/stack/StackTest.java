package q5b;

public class StackTest {

	MyStack stack;
	static final int NUM_STACKS = 2;
	int numToPop;
	int numToPush;
	
	public StackTest(int numToPush,  int numToPop) {
		this.numToPush = numToPush;
		this.numToPop = numToPop;
		stack = new LockFreeStack();
	}
	
	public void run() {
		Thread popper = new Thread(new Popper(numToPop));
		Thread pusher = new Thread(new Pusher(numToPush));
		
		
		pusher.start();
		popper.start();
		try
		{
			popper.join();
			pusher.join();
		}catch(Exception e){e.printStackTrace();}
		
		System.out.println(stack);
		//print size of stack
	}
	
	private class Popper implements Runnable{
		int numToPop;
		
		public Popper(int numToPop) {
			this.numToPop = numToPop;
		}
		@Override
		public void run() {
			// TODO Auto-generated method stub
			for(int i = 0; i < numToPop; i++)
			{
				try
				{
					stack.pop();
				}catch (EmptyStack e)
				{
					System.out.println("stack is empty");
				}
				
			}
		}
		
	}
	
	private class Pusher implements Runnable{
		int numToPush;
		public Pusher(int numToPush) {
			this.numToPush = numToPush;
		}
		@Override
		public void run() {
			// TODO Auto-generated method stub
			for(int i = 0; i < numToPush; i++)
			{
				stack.push(i);
			}
			try
			{
				stack.pop();
			}catch (EmptyStack e)
			{
				System.out.println("empty stack");
			}
			
		}
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		final int NUM_POP = 4;
		final int NUM_PUSH = 10;
		StackTest tester = new StackTest(NUM_PUSH, NUM_POP);
		tester.run();
	}

}
