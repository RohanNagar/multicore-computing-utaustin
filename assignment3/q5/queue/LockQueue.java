package q5;//ueue;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

public class LockQueue implements MyQueue {
// you are free to add members
ReentrantLock deqLock;
ReentrantLock enqLock;
Condition notEmpty;
AtomicInteger count;
Node tail;
Node head;

  public LockQueue() {
	// implement your constructor here
	  deqLock = new ReentrantLock();
	  enqLock = new ReentrantLock();
	  notEmpty = deqLock.newCondition();
	  count = new AtomicInteger();
	  Node sentinel = new Node(0);
	  head = sentinel;
	  tail = sentinel;
  }
  
  public boolean enq(Integer value) {
	// implement your enq method here
	  boolean signal = false;
	  enqLock.lock();
	  try
	  {
		  signal = head == tail;
		  
		  Node nodeToEnq = new Node(value);
		  tail.next = nodeToEnq;
		  tail = nodeToEnq;
		  count.incrementAndGet();
		  if(signal)
		  {
			  deqLock.lock();
			  try
			  {
				  notEmpty.signal();
			  }catch (Exception e)
 			  {
				  
			  }
			  finally
			  {
				  deqLock.unlock();
			  }
		  }
		  
	  } catch(OutOfMemoryError e)
	  {
		  return false;
	  }
	  finally
	  {
		  enqLock.unlock();
	  }
	  
    return true;
  }
  
  public Integer deq() {
	// implement your deq method here
	  int deqInt = 0;//value doesn't matter
	  deqLock.lock();
	  try
	  {
		  while(count.get() == 0)
		  {
			  notEmpty.await();
		  }
		  deqInt = head.next.value;
		  head = head.next;
		  
		  count.decrementAndGet();
	  }catch (Exception e)
	  {
		  e.printStackTrace(System.out);
	  }finally
	  {
		  deqLock.unlock();
	  }
	  
    return deqInt;
  }
  
  public boolean contains(Integer x)
  {
	  Node temp = head.next;
	  while(temp != null)
	  {
		  if(temp.value == x)
		  {
			  return true;
		  }
		  temp = temp.next;
	  }
	  return false;
  }
  
  public String toString()
  {
	  StringBuilder resultBuilder = new StringBuilder();
	  Node temp = head.next;
	  while(temp != null)
	  {
		  resultBuilder.append(temp.value + " ");
		  temp = temp.next;
	  }
	  return resultBuilder.toString();
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
