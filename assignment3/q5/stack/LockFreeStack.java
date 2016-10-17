package q5b;

import java.util.concurrent.atomic.AtomicReference;

import q5b.LockStack.Node;

public class LockFreeStack implements MyStack {
// you are free to add members
AtomicReference<Node> top;
	
  public LockFreeStack() {
	  // implement your constructor here
	  top = new AtomicReference<Node>();
  }
	
  public boolean push(Integer value) {
	  // implement your push method here
	  
	  Node nodeToPush = new Node(value);
	  try
	  {
		  while(true)
		  {
			  Node oldTop = top.get();
			  nodeToPush.next = oldTop;
			  if(top.compareAndSet(oldTop, nodeToPush))
			  {
				  return true;
			  }
			  else
			  {
				  Thread.yield();
			  }
		  }
	  }catch (OutOfMemoryError e)
	  {
		  return false;
	  }
	  
  }
  
  public Integer pop() throws EmptyStack {
	  // implement your pop method here
	  if(top.get() == null)
	  {
		  throw new EmptyStack();
	  }
	  while(true) {
		  Node oldTop = top.get();
		  if(top.compareAndSet(oldTop, oldTop.next))
		  {
			  return oldTop.value;
		  }
		  else
		  {
			  Thread.yield();
		  }
	  }
  }
  
//NOT THREAD-SAFE
  public String toString()
  {
	  
	  StringBuilder result = new StringBuilder();
	  Node n = top.get();
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
