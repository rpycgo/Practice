class List:    
    class ListNode:
        def __init__(self, data, _next):
            self.data = data
            self.next = _next
            
    def __init__(self):
        self.head = None
    
    def __getitem__(self, index):
        thru = self.head
        while (index > 0 and thru.next != None):
            index -= 1
            thru = thru.next
            
        return thru.data
    
    def insert_at_front(self, data):
        if self.head == None:
            self.head = self.ListNode(data, None)
        else:
            self.head = self.ListNode(data, self.head)
        
        
def main():
    
    lists = List()
    
    print('Inserting element 3 at front...')
    lists.insert_at_front(3)
    print(f'lists[0]: {lists[0]}')
    
    print('Inserting element 30 at front...')
    lists.insert_at_front(30)
    print(f'lists[0]: {lists[0]}')
    print(f'lists[1]: {lists[1]}')
    
    
    
if __name__ == '__main__':
    main()
