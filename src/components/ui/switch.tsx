import React from 'react';

type SwitchProps = {
  id?: string;
  checked?: boolean;
  onCheckedChange?: (checked: boolean) => void;
  className?: string;
};

export const Switch: React.FC<SwitchProps> = ({ id, checked = false, onCheckedChange, className }) => {
  return (
    <label className={`inline-flex items-center cursor-pointer ${className ?? ''}`} htmlFor={id}>
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(e) => onCheckedChange && onCheckedChange(e.target.checked)}
        className="sr-only"
      />
      <span
        aria-hidden
        className={`w-10 h-6 flex items-center rounded-full p-1 transition-colors duration-200 ${checked ? 'bg-blue-600' : 'bg-gray-300'}`}
      >
        <span
          className={`bg-white w-4 h-4 rounded-full shadow transform transition-transform duration-200 ${checked ? 'translate-x-4' : 'translate-x-0'}`}
        />
      </span>
    </label>
  );
};

export default Switch;
