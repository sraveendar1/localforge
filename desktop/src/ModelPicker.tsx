import { useState } from "react";

const MODEL_CATALOG: { [provider: string]: string[] } = {
  anthropic: ['claude-opus-5', 'claude-sonnet-5', 'claude-haiku-4-5-20251001', 'claude-fable-5-1'],
  openai: ['gpt-5'],
  gemini: ['gemini-2.5-pro'],
  local: ['ollama/llama3.1:70b', 'ollama/qwen2.5:72b']
};

const PROVIDER_LABELS: { [provider: string]: string } = {
  anthropic: 'Anthropic',
  openai: 'OpenAI',
  gemini: 'Gemini',
  local: 'Local (Ollama)'
};

const ALL_MODELS = Object.values(MODEL_CATALOG).flat();

export function ModelPicker({ value, onChange, disabled = false }: { value: string; onChange: (model: string) => void; disabled?: boolean }) {
  const [isOther, setIsOther] = useState(value !== '' && !ALL_MODELS.includes(value));
  const [otherModel, setOtherModel] = useState(value);

  const handleSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selected = e.target.value;
    if (selected === '__other__') {
      setIsOther(true);
    } else {
      setIsOther(false);
      onChange(selected);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setOtherModel(e.target.value);
  };

  const handleInputBlur = () => {
    onChange(otherModel.trim());
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      onChange(otherModel.trim());
    }
  };

  return (
    <div className="flex items-center gap-2">
      <select
        value={isOther ? '__other__' : (value || 'claude-opus-5')}
        onChange={handleSelectChange}
        disabled={disabled}
        className={`block w-full rounded-sm border border-mx-dim bg-mx-panel2 px-2 py-1 text-sm ${
          disabled ? 'cursor-not-allowed opacity-50' : ''
        }`}
      >
        {Object.entries(MODEL_CATALOG).map(([provider, models]) => (
          <optgroup key={provider} label={PROVIDER_LABELS[provider] ?? provider}>
            {models.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </optgroup>
        ))}
        <option value="__other__">Other…</option>
      </select>
      {isOther && (
        <input
          type="text"
          value={otherModel}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          onKeyDown={handleInputKeyDown}
          disabled={disabled}
          className={`block w-full rounded-sm border border-mx-dim bg-mx-panel2 px-2 py-1 text-sm ${
            disabled ? 'cursor-not-allowed opacity-50' : ''
          }`}
        />
      )}
    </div>
  );
}
