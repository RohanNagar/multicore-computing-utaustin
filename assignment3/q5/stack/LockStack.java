package q5b;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class LockStack implements MyStack {
// you are free to add members
Node top;
ReentrantLock topLock;
AtomicInteger count;

  public LockStack() {
	  // implement your constructor here
	  topLock = new ReentrantLock();
	  top = null;
	  count = new AtomicInteger(0);
  }
  
  public boolean push(Integer value) {
	  // implement your push method here
	  Node nodeToPush = new Node(value);
	  topLock.lock();
	  try
	  {
		  nodeToPush.next = top;
		  top = nodeToPush;
		  count.incrementAndGet();
		  //condition.signal();
		  
	  }catch(OutOfMemoryError e)
	  {
		  return false;
	  }
	  catch (Exception e){e.printStackTrace(System.out);}
	  finally
	  {
		  topLock.unlock();
	  }
	  return true;
  }
  
  public Integer pop() throws EmptyStack {
	  // implement your pop method here
	  Node nodeToDeq = null;
	  topLock.lock();
	  try
	  {
		  if(top == null)
		  {
			  throw new EmptyStack();
		  }
		  nodeToDeq = top;
		  top = top.next;
		  count.decrementAndGet();
	  } finally
	  {
		  topLock.unlock();
	  }
	  
	  return nodeToDeq.value;
  }
  
  //NOT THREAD-SAFE
  public String toString()
  {
	  
	  StringBuilder result = new StringBuilder();
	  Node n = top;
	  while(n != null)
	  {
		  result.append(n.value + " ");
		  n = n.next;
	  }
	  return result.toString();
  }
  
  protected class Node {
	  public Integer value;
	  public Node next;
		    
	  public Node(Integer x) {
		  value = x;
		  next = null;
	  }
  }
}
