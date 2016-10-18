package q4;

import java.util.concurrent.locks.ReentrantLock;

public class FineGrainedListSet implements ListSet {
  private Node head;
	
  public FineGrainedListSet() {
    head = new Node(null);
  }
	  
  public boolean add(int value) {
    Node newNode = new Node(value);
    Node current = head;

    current.lock.lock();
    while (current.next != null) {
      // current is locked
      current.next.lock.lock();

      if (value <= current.next.value) {
        break;
      }

      Node temp = current.next;
      current.lock.unlock();
      current = temp;
    }

    newNode.next = current.next;
    current.next = newNode;

    if (current.next.next != null) {
      current.next.next.lock.unlock();
    }

    current.lock.unlock();
    return true;
  }
	  
  public boolean remove(int value) {
    Node pred = head;
    pred.lock.lock();

    Node current = pred.next;
    current.lock.lock();

    // Find it
    while (current.value < value) {
      pred.lock.unlock();
      pred = current;
      current = current.next;
      current.lock.lock();
    }

    // If we found it, remove it and return
    if (current.value == value) {
      pred.next = current.next;

      pred.lock.unlock();
      current.lock.unlock();
      return true;
    }

    // Make sure to unlock before returning
    pred.lock.unlock();
    current.lock.unlock();
    return false;
  }
	  
  public boolean contains(int value) {
    Node current = head;

    current.lock.lock();
    while (current != null) {
      if (current.value != null && current.value == value) {
        current.lock.unlock();
        return true;
      }

      if (current.value != null && current.value > value) {
        current.lock.unlock();
        return false;
      }

      Node temp = current.next;
      if (temp != null) {
        temp.lock.lock();
      }

      current.lock.unlock();
      current = temp;
    }

    return false;
  }
	  
  protected class Node {
    public Integer value;
    public ReentrantLock lock;
	  public Node next;
			    
    public Node(Integer x) {
      value = x;
      lock = new ReentrantLock();
      next = null;
    }
  }

//  public static void main(String[] args) {
//    FineGrainedListSet list = new FineGrainedListSet();
//
//    Thread t1 = new Thread() {
//      @Override
//      public void run() {
//        list.add(3);
//        list.add(7);
//        list.add(8);
//        list.add(10);
//        list.add(11);
//        list.add(235);
//
//        list.remove(5);
//
//        System.out.println(list.contains(3));
//      }
//    };
//
//    Thread t2 = new Thread() {
//      @Override
//      public void run() {
//        list.add(16);
//        list.add(5);
//
//        System.out.println(list.contains(5));
//      }
//    };
//
//    t1.start();
//    t2.start();
//  }
}
