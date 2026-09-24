import React from 'react';

interface StatusBarProps {
  folder: string | null;
  connected: boolean;
  running: boolean;
  model: string;
  autoApprove: boolean;
  todoCount: number;
  doneCount: number;
  onCancel: () => void;
}

const StatusBar: React.FC<StatusBarProps> = ({ folder, connected, running, model, autoApprove, todoCount, doneCount, onCancel }) => {
  const statusText = running ? 'working' : connected ? 'ready' : 'offline';
  const folderSegment = folder ? folder.split('/').pop() || 'no folder' : '';
  const progressText = todoCount > 0 ? `${doneCount}/${todoCount}` : '';
  const statusClass = running ? 'bg-mx-amber' : connected ? 'bg-mx-green' : 'bg-mx-dim';
  const wordClass = running ? 'text-mx-amber' : connected ? 'text-mx-green' : 'text-mx-dim';

  return (
    <div className="flex items-center gap-3 shrink-0 overflow-hidden border-t border-mx-dim bg-mx-panel px-3 py-1 text-[11px]">
      <span className={`inline-block h-2 w-2 rounded-full ${statusClass} ${running ? 'animate-pulse' : ''}`}></span>
      <span className={wordClass}>{statusText}</span>
      <span className="text-mx-dim">|</span>
      <span className="text-mx-mid" title={folder || ''}>{folderSegment || 'no folder'}</span>
      <span className="text-mx-dim">|</span>
      <span className="text-mx-dim">model</span>
      <span className="text-mx-mid">{model}</span>
      <span className="text-mx-dim">|</span>
      <span className={autoApprove ? 'text-mx-green' : 'text-mx-amber'}>{autoApprove ? 'auto' : 'manual'}</span>
      {progressText && <><span className="text-mx-dim">|</span><span className="text-mx-dim">todos</span><span className="text-mx-mid">{progressText}</span></>}
      {running && <button onClick={onCancel} className="ml-auto rounded-sm border border-mx-red bg-transparent px-2 text-mx-red hover:border-mx-bright hover:text-mx-bright">Cancel</button>}
    </div>
  );
};

export { StatusBar };
