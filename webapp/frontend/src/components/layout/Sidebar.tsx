import { NavLink } from 'react-router-dom';

const navItems = [
  { to: '/', label: 'Dashboard', icon: '📊' },
  { to: '/sentiment', label: 'Sentiment', icon: '💬' },
  { to: '/topics', label: 'Topics', icon: '🏷️' },
  { to: '/clusters', label: 'Clusters', icon: '🔮' },
];

export default function Sidebar() {
  return (
    <aside className="w-56 bg-slate-900 text-white min-h-screen p-4 flex flex-col">
      <h1 className="text-lg font-bold mb-1">Venezuela-US</h1>
      <p className="text-xs text-slate-400 mb-6">Narrative Analysis</p>
      <nav className="flex flex-col gap-1">
        {navItems.map(item => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `px-3 py-2 rounded text-sm transition ${
                isActive
                  ? 'bg-slate-700 text-white font-medium'
                  : 'text-slate-300 hover:bg-slate-800'
              }`
            }
          >
            <span className="mr-2">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto pt-4 border-t border-slate-700">
        <p className="text-xs text-slate-500">Platforms: Reddit</p>
        <p className="text-xs text-slate-500">TikTok, GDELT (coming)</p>
      </div>
    </aside>
  );
}
