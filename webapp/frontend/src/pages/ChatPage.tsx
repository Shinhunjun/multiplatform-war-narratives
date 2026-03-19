import { useState, useRef, useEffect } from 'react';
import { sendChatMessage } from '../lib/api';
import type { ChatMessage } from '../lib/api';

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || loading) return;

    const userMsg: ChatMessage = { role: 'user', content: q };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const { answer } = await sendChatMessage(q, messages);
      setMessages(prev => [...prev, { role: 'assistant', content: answer }]);
    } catch (e: any) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${e?.message || e}` }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="px-6 py-8 flex flex-col" style={{ height: 'calc(100vh - 48px)' }}>
      <div className="mb-4">
        <h2 className="text-xl font-bold text-[#e8eaed] tracking-tight">Data Chat</h2>
        <p className="text-[12px] text-[#8b8fa3] mt-1">Ask questions about Venezuela-US discourse across all platforms</p>
        <div className="h-[2px] w-10 bg-[#38bdf8] mt-2 rounded-full" />
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 min-h-0 pb-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <p className="text-[#8b8fa3] text-sm mb-4">Ask anything about the data. For example:</p>
              <div className="space-y-2">
                {[
                  '2024년 선거 기간에 Reddit과 TikTok의 감성 차이는?',
                  'Which platform has the most negative sentiment?',
                  'Top TikTok 해시태그는 뭐야?',
                  'Compare topic trends across platforms',
                ].map((q, i) => (
                  <button
                    key={i}
                    onClick={() => { setInput(q); }}
                    className="block w-full text-left px-4 py-2.5 rounded-lg border border-[#2a2e3d] bg-[#1a1d27] text-[13px] text-[#c4c8d8] hover:border-[#38bdf8] hover:text-[#e8eaed] transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[75%] rounded-lg px-4 py-3 text-[13px] leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-[#38bdf8]/15 text-[#e8eaed] border border-[#38bdf8]/30'
                  : 'bg-[#1a1d27] text-[#c4c8d8] border border-[#2a2e3d]'
              }`}
            >
              {msg.role === 'assistant' ? (
                <div
                  dangerouslySetInnerHTML={{
                    __html: msg.content
                      .replace(/\*\*(.+?)\*\*/g, '<strong class="text-[#e8eaed]">$1</strong>')
                      .replace(/\n/g, '<br/>')
                  }}
                />
              ) : (
                <p className="whitespace-pre-wrap">{msg.content}</p>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-[#1a1d27] border border-[#2a2e3d] rounded-lg px-4 py-3">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-[#38bdf8] rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-[#38bdf8] rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-[#38bdf8] rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-[#2a2e3d] pt-4">
        <div className="flex gap-3">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about Venezuela-US discourse..."
            rows={1}
            className="flex-1 bg-[#1a1d27] border border-[#2a2e3d] rounded-lg px-4 py-2.5 text-[13px] text-[#e8eaed] placeholder-[#64748b] resize-none focus:border-[#38bdf8] outline-none"
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-5 py-2.5 bg-[#38bdf8] text-[#0f1117] rounded-lg text-[13px] font-semibold hover:bg-[#7dd3fc] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
