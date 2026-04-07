import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import {
  FileUp, Send, Bot, User, Loader2, FileText,
  CheckCircle, AlertCircle, Copy, RefreshCw,
  HelpCircle, ChevronRight
} from 'lucide-react'
import './App.css'

let rawApiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'
rawApiUrl = rawApiUrl.replace(/\/+$/, '')
if (!rawApiUrl.endsWith('/api')) {
  rawApiUrl += '/api'
}
const API_BASE_URL = rawApiUrl
console.log("Frontend connecting to:", API_BASE_URL)

const SUGGESTIONS = [
  "Summarize this document",
  "Key takeaways",
  "Find information about...",
  "Main dates and events"
]

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [streamingMessage, setStreamingMessage] = useState('')

  const [file, setFile] = useState(null)
  const [uploadStatus, setUploadStatus] = useState('idle')
  
  const [repoPath, setRepoPath] = useState('')
  const [repoStatus, setRepoStatus] = useState('idle')

  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)
  const chatContainerRef = useRef(null)

  const scrollToBottom = () => {
    if (!messagesEndRef.current || !chatContainerRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'auto' })
      return
    }

    const container = chatContainerRef.current
    const isAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 150

    if (isAtBottom) {
      messagesEndRef.current.scrollIntoView({ behavior: 'auto' })
    }
  }

  useEffect(() => {
    const timer = setTimeout(scrollToBottom, 50)
    return () => clearTimeout(timer)
  }, [messages, isTyping, streamingMessage])

  // Proactive warm-up ping for Render backend
  useEffect(() => {
    const warmup = async () => {
      try {
        const healthUrl = API_BASE_URL.replace(/\/api$/, '') + '/health'
        await axios.get(healthUrl)
        console.log("Backend connection established")
      } catch (err) {
        console.warn("Warm-up ping failed:", err.message)
      }
    }
    warmup()
  }, [])

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0]
    if (!selectedFile) return
    if (!selectedFile.name.endsWith('.pdf')) {
      alert("Please upload a PDF file.")
      return
    }

    setFile(selectedFile)
    setUploadStatus('uploading')

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      await axios.post(`${API_BASE_URL}/upload`, formData)
      setUploadStatus('success')
      setMessages(prev => [...prev, {
        role: 'system',
        content: `Document analyzed: ${selectedFile.name}`
      }])
    } catch (err) {
      console.error("Upload error details:", err)
      const errorMsg = err.response?.data?.detail || err.message
      setUploadStatus('error')
      console.error("Upload error:", error)
    }
  }

  const handleAnalyzeRepo = async (e) => {
    e?.preventDefault()
    if (!repoPath.trim()) return

    setRepoStatus('uploading')
    try {
      await axios.post(`${API_BASE_URL}/analyze-repo`, { path: repoPath })
      setRepoStatus('success')
      setMessages(prev => [...prev, {
        role: 'system',
        content: `Repository analyzed: ${repoPath}`
      }])
      setFile({ name: repoPath.split('/').pop() || 'Repository' })
      setRepoPath('')
    } catch (err) {
      console.error("Repo analysis error details:", err)
      const errorMsg = err.response?.data?.detail || err.message
      setRepoStatus('error')
      alert(`Error analyzing repository: ${errorMsg}`)
    }
  }

  const simulateStreaming = async (text) => {
    const words = text.split(' ')
    let currentText = ''
    for (let i = 0; i < words.length; i++) {
      currentText += words[i] + ' '
      setStreamingMessage(currentText)
      await new Promise(resolve => setTimeout(resolve, 30 + Math.random() * 40))
    }
    setMessages(prev => [...prev, { role: 'ai', content: text }])
    setStreamingMessage('')
  }

  const handleSend = async (queryText) => {
    const query = queryText || input
    if (!query.trim() || isTyping) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: query }])
    setIsTyping(true)

    const fullUrl = `${API_BASE_URL}/ask`
    try {
      const response = await axios.post(fullUrl, { query })
      setIsTyping(false)
      await simulateStreaming(response.data.answer)
    } catch (error) {
      setIsTyping(false)
      const msg = error.response ? ` (Status: ${error.response.status})` : ` (${error.message})`
      setMessages(prev => [...prev, {
        role: 'ai',
        content: `Error connecting to ${fullUrl}${msg}. Please check the VITE_API_URL in Render.`
      }])
      console.error("Connection error at:", fullUrl, error)
    }
  }

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text)
  }

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>AskMyDocs</h1>
        </div>

        <div className="upload-container">
          <p className="upload-info">
            Upload a PDF to search and analyze its content.
          </p>

          <input type="file" accept=".pdf" className="file-input" ref={fileInputRef} onChange={handleFileChange} />

          <button
            className={`upload-btn ${uploadStatus === 'uploading' ? 'uploading' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadStatus === 'uploading'}
          >
            {uploadStatus === 'uploading' ? <Loader2 className="spinner" size={20} /> : <FileUp size={20} />}
            {uploadStatus === 'uploading' ? 'Analyzing...' : 'Upload PDF'}
          </button>

          <div style={{ marginTop: '2.5rem', paddingTop: '2.5rem', borderTop: '1px solid var(--border-color)' }}>
            <h2 style={{ fontSize: '0.85rem', color: 'var(--accent-color)', marginBottom: '1.25rem', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Explore Repository</h2>
            <form onSubmit={handleAnalyzeRepo} theme-ignore="true" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <input 
                type="text" 
                value={repoPath}
                autoComplete="off"
                onChange={(e) => setRepoPath(e.target.value)}
                placeholder="Local path or GitHub URL" 
                style={{
                  background: 'var(--bg-color)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius)',
                  padding: '1rem 1.4rem',
                  color: 'var(--text-primary)',
                  fontSize: '0.95rem',
                  outline: 'none',
                  transition: 'var(--transition)'
                }}
              />
              <button 
                type="submit"
                disabled={!repoPath.trim() || repoStatus === 'uploading'}
                className={`upload-btn ${repoStatus === 'uploading' ? 'uploading' : ''}`}
                style={{
                  width: '100%',
                  background: 'linear-gradient(135deg, #6c5ce7, #a29bfe)', // A subtle blue/purple that fits but distinguishes from PDF
                  boxShadow: '0 8px 20px rgba(108, 92, 231, 0.2)'
                }}
              >
                {repoStatus === 'uploading' ? <Loader2 className="spinner" size={20} /> : <RefreshCw size={20} />}
                {repoStatus === 'uploading' ? 'Scanning...' : 'Analyze Repo'}
              </button>
            </form>
          </div>

          {file && (
            <div className="file-status-card">
              <span className="label">Active File</span>
              <div className={`file-status ${uploadStatus}`}>
                {uploadStatus === 'success' ? <CheckCircle size={16} /> : <FileText size={16} />}
                <span className="truncate">{file.name}</span>
              </div>
            </div>
          )}
        </div>
      </aside>

      <main className="main-content">
        <div className="chat-container">
          <div className="messages" ref={chatContainerRef}>
            {messages.length === 0 ? (
              <div className="empty-state">
                <h2>No active conversation</h2>
                <p>Upload a document and ask a question to begin.</p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className="message-meta">
                    {msg.role === 'user' ? <User size={14} /> : <Bot size={14} />}
                    <span>{msg.role === 'user' ? 'You' : 'Assistant'}</span>
                  </div>
                  <div className="message-bubble">
                    <div className="message-content" style={{ whiteSpace: 'pre-wrap' }}>
                      {msg.content}
                    </div>
                  </div>
                  {msg.role === 'ai' && (
                    <div className="message-actions">
                      <button className="action-btn" onClick={() => copyToClipboard(msg.content)} title="Copy"><Copy size={16} /></button>
                      <button className="action-btn" onClick={() => handleSend(messages[idx - 1]?.content)} title="Retry"><RefreshCw size={16} /></button>
                    </div>
                  )}
                </div>
              ))
            )}

            {streamingMessage && (
              <div className="message ai">
                <div className="message-meta"><Bot size={14} /><span>Assistant</span></div>
                <div className="message-bubble">
                  <div className="message-content" style={{ whiteSpace: 'pre-wrap' }}>
                    {streamingMessage}
                  </div>
                </div>
              </div>
            )}


            {isTyping && !streamingMessage && (
              <div className="message ai">
                <div className="typing-indicator">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-wrapper">
            {file && messages.length < 5 && (
              <div className="suggestions">
                {SUGGESTIONS.map((s, i) => (
                  <div key={i} className="suggestion-chip" onClick={() => handleSend(s)}>
                    {s} <ChevronRight size={14} />
                  </div>
                ))}
              </div>
            )}

            <form className="input-area" onSubmit={(e) => { e.preventDefault(); e.stopPropagation(); handleSend(); }}>
              <input
                type="text" value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask me anything..."
                disabled={isTyping}
              />
              <button type="submit" disabled={!input.trim() || isTyping}>
                <Send size={18} />
              </button>
            </form>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
