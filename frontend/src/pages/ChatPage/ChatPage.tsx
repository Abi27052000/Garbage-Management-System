import { useState, useEffect, useRef, useCallback } from "react";
import "./ChatPage.css";
import {
  getCurrentUser,
  getChatUsers,
  getChatHistory,
  getAllUsers,
} from "../../utility/api";
import { FaCheck, FaCheckDouble } from "react-icons/fa";
import { io, Socket } from "socket.io-client";

interface User {
  _id: string;
  username: string;
  email: string;
  isOnline: boolean;
  lastSeen: Date;
  lastMessageAt?: string;
  lastMessage?: string;
}

interface Message {
  _id: string;
  sender: { _id: string; username: string } | string;
  receiver: { _id: string; username: string } | string;
  message: string;
  isSeen: boolean;
  createdAt: string;
}

const SOCKET_URL = "http://localhost:3000";

const ChatPage = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState("");
  const [currentUser, setCurrentUser] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [allUsers, setAllUsers] = useState<User[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [isTyping, setIsTyping] = useState(false);

  const socketRef = useRef<Socket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const selectedUserRef = useRef<User | null>(null);
  const currentUserRef = useRef<any>(null);
  const typingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep refs in sync
  useEffect(() => { selectedUserRef.current = selectedUser; }, [selectedUser]);
  useEffect(() => { currentUserRef.current = currentUser; }, [currentUser]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => { scrollToBottom(); }, [messages]);

  // Helper to get sender id regardless of populated or not
  const getSenderId = (msg: Message) => {
    if (!msg.sender) return null;
    return typeof msg.sender === "object" ? msg.sender._id : msg.sender;
  };

  const fetchUsers = useCallback(async (userId: string) => {
    const data = await getChatUsers(userId);
    if (data.success) setUsers(data.contacts);
  }, []);

  const fetchAllUsers = useCallback(async (userId: string) => {
    const data = await getAllUsers(userId);
    if (data.success) setAllUsers(data.users);
  }, []);

  const fetchMessages = useCallback(async (userId: string) => {
    const cu = currentUserRef.current;
    if (!cu) return;
    const data = await getChatHistory(userId, cu.id);
    if (data.success) setMessages(data.messages);
  }, []);

  // Initialize user + socket
  useEffect(() => {
    const user = getCurrentUser();
    if (!user) return;
    setCurrentUser(user);
    fetchUsers(user.id);
    fetchAllUsers(user.id);

    // Setup socket
    const socket = io(SOCKET_URL, { transports: ["websocket"] });
    socketRef.current = socket;

    socket.on("connect", () => {
      socket.emit("join", user.id);
    });

    // Real-time incoming message
    socket.on("receive_message", (msg: Message) => {
      const su = selectedUserRef.current;
      const cu = currentUserRef.current;
      const senderId = getSenderId(msg);

      // Add message if the chat window matches
      if (su && senderId === su._id) {
        setMessages((prev) => {
          // avoid duplicates
          if (prev.some((m) => m._id === msg._id)) return prev;
          return [...prev, msg];
        });
        // Mark as seen immediately
        if (cu) {
          socket.emit("mark_seen", { senderId: su._id, receiverId: cu.id });
        }
      }
      // Refresh contact list for last message preview
      if (cu) fetchUsers(cu.id);
    });

    // Confirmation that our sent message was saved
    socket.on("message_sent", (msg: Message) => {
      setMessages((prev) => {
        if (prev.some((m) => m._id === msg._id)) return prev;
        return [...prev, msg];
      });
      if (currentUserRef.current) fetchUsers(currentUserRef.current.id);
    });

    // Online status
    socket.on("user_online", ({ userId }: { userId: string }) => {
      setUsers((prev) =>
        prev.map((u) => (u._id === userId ? { ...u, isOnline: true } : u))
      );
      setAllUsers((prev) =>
        prev.map((u) => (u._id === userId ? { ...u, isOnline: true } : u))
      );
      setSelectedUser((prev) =>
        prev && prev._id === userId ? { ...prev, isOnline: true } : prev
      );
    });

    // Offline status
    socket.on("user_offline", ({ userId, lastSeen }: { userId: string; lastSeen: Date }) => {
      setUsers((prev) =>
        prev.map((u) => (u._id === userId ? { ...u, isOnline: false, lastSeen } : u))
      );
      setAllUsers((prev) =>
        prev.map((u) => (u._id === userId ? { ...u, isOnline: false, lastSeen } : u))
      );
      setSelectedUser((prev) =>
        prev && prev._id === userId ? { ...prev, isOnline: false, lastSeen } : prev
      );
    });

    // Typing indicator
    socket.on("user_typing", ({ isTyping: typing }: { userId: string; isTyping: boolean }) => {
      setIsTyping(typing);
    });

    // Messages seen acknowledgement
    socket.on("messages_seen", () => {
      setMessages((prev) => prev.map((m) => ({ ...m, isSeen: true })));
    });

    return () => {
      socket.disconnect();
    };
  }, [fetchUsers, fetchAllUsers]);

  // Load messages when selected user changes
  useEffect(() => {
    if (selectedUser && currentUser) {
      fetchMessages(selectedUser._id);
      // Mark their messages as seen
      if (socketRef.current) {
        socketRef.current.emit("mark_seen", {
          senderId: selectedUser._id,
          receiverId: currentUser.id,
        });
      }
    }
  }, [selectedUser, currentUser, fetchMessages]);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const query = e.target.value;
    setSearchQuery(query);
    setShowDropdown(query.trim() !== "");
  };

  const handleTyping = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNewMessage(e.target.value);
    if (socketRef.current && selectedUser && currentUser) {
      socketRef.current.emit("typing", { receiverId: selectedUser._id, isTyping: true });
      if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
      typingTimeoutRef.current = setTimeout(() => {
        socketRef.current?.emit("typing", { receiverId: selectedUser._id, isTyping: false });
      }, 1500);
    }
  };

  const sendMessageHandler = () => {
    if (!newMessage.trim() || !selectedUser || !currentUser) return;
    socketRef.current?.emit("send_message", {
      receiverId: selectedUser._id,
      message: newMessage.trim(),
    });
    // Stop typing indicator
    socketRef.current?.emit("typing", { receiverId: selectedUser._id, isTyping: false });
    setNewMessage("");
  };

  return (
    <div className="chat-page">
      <div className="chat-container">
        <div className="users-list">
          <h3>Users</h3>
          <div className="search-container">
            <input
              type="text"
              placeholder="Search users by username or email..."
              value={searchQuery}
              onChange={handleSearchChange}
              className="search-input"
            />
            {showDropdown && (
              <div className="search-dropdown">
                {allUsers
                  .filter(
                    (user) =>
                      currentUser &&
                      user._id !== currentUser.id &&
                      (user.username.toLowerCase().includes(searchQuery.toLowerCase()) ||
                        user.email.toLowerCase().includes(searchQuery.toLowerCase()))
                  )
                  .map((user) => (
                    <div
                      key={user._id}
                      className="dropdown-item"
                      onClick={() => {
                        setSelectedUser(user);
                        setSearchQuery("");
                        setShowDropdown(false);
                      }}
                    >
                      <div className="user-info">
                        <span className="username">{user.username}</span>
                        <span className={`status ${user.isOnline ? "online" : "offline"}`}></span>
                      </div>
                    </div>
                  ))}
              </div>
            )}
          </div>
          <ul>
            {users.map((user) => (
              <li
                key={user._id}
                className={selectedUser?._id === user._id ? "active" : ""}
                onClick={() => setSelectedUser(user)}
              >
                <div className="user-info">
                  <span className="username">{user.username}</span>
                  <span className={`status ${user.isOnline ? "online" : "offline"}`}></span>
                </div>
                {user.lastMessage && (
                  <div className="last-message">
                    <span className="last-msg-text">{user.lastMessage}</span>
                    {user.lastMessageAt && (
                      <span className="last-msg-time">
                        {new Date(user.lastMessageAt).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
        <div className="chat-window">
          {selectedUser ? (
            <>
              <div className="chat-header">
                <div>
                  <h3>Chat with {selectedUser.username}</h3>
                  {isTyping ? (
                    <span className="typing-indicator">typing...</span>
                  ) : (
                    <span className={`header-status ${selectedUser.isOnline ? "online-text" : "offline-text"}`}>
                      {selectedUser.isOnline ? "Online" : "Offline"}
                    </span>
                  )}
                </div>
              </div>
              <div className="messages">
                {messages.map((msg) => {
                  const senderId = getSenderId(msg);
                  const isMine = senderId === currentUser?.id;
                  return (
                    <div key={msg._id} className={`message ${isMine ? "sent" : "received"}`}>
                      <p>{msg.message}</p>
                      <div className="message-footer">
                        <span className="timestamp">
                          {new Date(msg.createdAt).toLocaleTimeString()}
                        </span>
                        {isMine && (
                          msg.isSeen
                            ? <FaCheckDouble className="tick seen" />
                            : <FaCheck className="tick unseen" />
                        )}
                      </div>
                    </div>
                  );
                })}
                <div ref={messagesEndRef} />
              </div>
              <div className="message-input">
                <input
                  type="text"
                  value={newMessage}
                  onChange={handleTyping}
                  placeholder="Type a message..."
                  onKeyDown={(e) => e.key === "Enter" && sendMessageHandler()}
                />
                <button onClick={sendMessageHandler}>Send</button>
              </div>
            </>
          ) : (
            <div className="no-chat">Select a user to start chatting</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatPage;
