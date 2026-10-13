// The LocalForge mark: an anvil with a spark, drawn in the app's own greens so it needs no image file.
export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" role="img" aria-label="LocalForge" data-testid="logo" className="shrink-0">
      <rect x="1" y="1" width="30" height="30" rx="7" fill="#050a05" stroke="#27a35b" strokeWidth="1.5" />
      <path d="M6 13h15.5c1.6 0 3-.6 4.5-2v4.2c-1.4 1.3-3 2-5 2.1-.3 1.6-1.2 2.7-2.6 3.4V22H21v3H9v-3h3.6v-1.3C10.8 20 9.6 18.700 9 17H6z" fill="#00ff41" />
      <path d="M23.500 3.500l.9 2.600 2.600.9-2.600.9-.9 2.600-.9-2.600-2.600-.9 2.600-.9z" fill="#6bff9a" />
    </svg>
  );
}
