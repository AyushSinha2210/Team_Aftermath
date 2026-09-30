import './controls.css';

export function Button({ className = '', children, ...props }) {
  return <button className={`button ${className}`} {...props}>{children}</button>;
}

export function QueryInput({ error, ...props }) {
  return <input className="query-input" aria-invalid={Boolean(error)} {...props} />;
}
