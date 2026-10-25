import { useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";

/** The largest box of the given aspect ratio (width / height) that fits the available space, centred. */
export function FitBox({ aspect, children, style }: { aspect: number; children: ReactNode; style?: CSSProperties }) {
  const outer = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  useLayoutEffect(() => {
    const el = outer.current!;
    const fit = () => {
      const { clientWidth: W, clientHeight: H } = el;
      const w = Math.min(W, H * aspect);
      setSize({ w, h: w / aspect });
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
  }, [aspect]);

  return (
    <div ref={outer} className="fit-outer">
      <div className="media" style={{ width: size.w, height: size.h, ...style }}>{children}</div>
    </div>
  );
}
