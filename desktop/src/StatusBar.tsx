import React from 'react';

interface StatusBarProps {
  folder: string | null;
  connected: boolean;
  running: boolean;
  model: string;
  version?: string;
  autoApprove: boolean;
  todoCount: number;
  doneCount: number;
  onCancel: () => void;
}

const StatusBar: React.FC<StatusBarProps> = ({ folder, connected, running, model, version, autoApprove, todoCount, doneCount, onCancel }) => {
  const statusText = running ? 'working' : connected ? 'ready' : 'offline';
  const folderSegment = folder ? folder.split('/').pop() || 'no folder' : '';
  const progressText = todoCount > 0 ? `${doneCount}/${todoCount}` : '';
  const statusClass = running ? 'bg-mx-amber' : connected ? 'bg-mx-green' : 'bg-mx-dim';
  const wordClass = running ? 'text-mx-amber' : connected ? 'text-mx-green' : 'text-mx-muted';

  return (
    <div className="flex items-center gap-3 shrink-0 overflow-hidden border-t border-mx-dim bg-mx-panel px-3 py-1 text-[11px]">
      <span className={`inline-block h-2 w-2 rounded-full ${statusClass} ${running ? 'animate-pulse' : ''}`}></span>
      <span className={wordClass}>{statusText}</span>
      <span className="text-mx-muted">|</span>
      <span className="text-mx-mid" title={folder || ''}>{folderSegment || 'no folder'}</span>
      <span className="text-mx-muted">|</span>
      <span className="text-mx-muted">model</span>
      <span className="text-mx-mid">{model}</span>
      <span className="text-mx-muted">|</span>
      <span className={autoApprove ? 'text-mx-green' : 'text-mx-amber'}>{autoApprove ? 'auto' : 'manual'}</span>
      {progressText && <><span className="text-mx-muted">|</span><span className="text-mx-muted">todos</span><span className="text-mx-mid">{progressText}</span></>}
      {version && <span className="ml-auto text-mx-muted" title="LocalForge version (yyyymmdd.N)" data-testid="app-version">v{version}</span>}
      {running && <button onClick={onCancel} className="ml-2 rounded-sm border border-mx-red bg-transparent px-2 text-mx-red hover:border-mx-bright hover:text-mx-bright">Cancel</button>}
    </div>
  );
};

export { StatusBar };
