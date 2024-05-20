import { SVGAttributes, useEffect, useState } from "react";

export type IconComponent = React.FC<SVGAttributes<SVGSVGElement>>;

export type SelectableIconComponent = React.FC<
  SVGAttributes<SVGSVGElement> & { selected?: boolean }
>;

export const Plus: SelectableIconComponent = ({
  selected = false,
  className,
  ...props
}) => {
  return (
    <svg
      viewBox="0 0 100 100"
      width="100"
      stroke={"black"}
      strokeWidth={8}
      className={`transition-all duration-500 ${
        selected ? "rotate-45" : ""
      } ${className}`}
      {...props}
    >
      <line {...{ x1: "50", y1: "10", x2: "50", y2: "90" }} />
      <line {...{ x1: "10", y1: "50", x2: "90", y2: "50" }} />
    </svg>
  );
};

export const TriangleDown: SelectableIconComponent = ({
  selected = false,
  ...props
}) => {
  const [hasBeenSelected, setHasBeenSelected] = useState(false);

  useEffect(() => {
    if (selected && !hasBeenSelected) {
      setHasBeenSelected(true);
    }
  }, [selected, hasBeenSelected]);

  return (
    <svg
      viewBox="0 0 100 100"
      fill="none"
      stroke="black"
      strokeWidth={8}
      strokeLinecap="round"
      {...props}
    >
      <line
        style={{ transformOrigin: "10px 25px" }}
        className={`${
          selected
            ? "animate-downTriangle_TopLine_Forwards"
            : hasBeenSelected
            ? "animate-downTriangle_TopLine_Reverse"
            : ""
        }`}
        {...{ x1: "10", y1: "25", x2: "90", y2: "25" }}
      />
      <line
        style={{ transformOrigin: "50px 95px" }}
        className={`${
          selected
            ? "animate-downTriangle_LeftLine_Forwards"
            : hasBeenSelected
            ? "animate-downTriangle_LeftLine_Reverse"
            : ""
        }`}
        {...{ x1: "50", y1: "95", x2: "10", y2: "25" }}
      />
      <line
        style={{ transformOrigin: "50px 95px" }}
        className={`${
          selected
            ? "animate-downTriangle_RightLine_Forwards"
            : hasBeenSelected
            ? "animate-downTriangle_RightLine_Reverse"
            : ""
        }`}
        {...{ x1: "90", y1: "25", x2: "50", y2: "95" }}
      />
    </svg>
  );
};

export const HorizontalLines: React.FC<SVGAttributes<SVGSVGElement>> = ({
  ...props
}) => {
  return (
    <svg viewBox="0 0 100 100" stroke="black" strokeWidth={8} {...props}>
      <line x1="10" y1="20" x2="90" y2="20" />
      <line x1="10" y1="40" x2="90" y2="40" />
      <line x1="10" y1="60" x2="90" y2="60" />
      <line x1="10" y1="80" x2="90" y2="80" />
    </svg>
  );
};

export const Circle: React.FC<SVGAttributes<SVGSVGElement>> = ({
  ...props
}) => {
  return (
    <svg
      viewBox="0 0 100 100"
      fill="none"
      stroke="black"
      strokeWidth={8}
      {...props}
    >
      <circle cx="50" cy="50" r="45" />
    </svg>
  );
};

export const Star: React.FC<SVGAttributes<SVGSVGElement>> = ({ ...props }) => {
  return (
    <svg
      viewBox="0 0 100 100"
      stroke="black"
      strokeWidth={8}
      fill="none"
      {...props}
    >
      <g>
        <line x1="10" y1="50" x2="90" y2="50" />
        <line x1="50" y1="10" x2="50" y2="90" />
        <line x1="25" y1="25" x2="75" y2="75" />
        <line x1="75" y1="25" x2="25" y2="75" />
      </g>
    </svg>
  );
};

export const BackChevrons: React.FC<SVGAttributes<SVGSVGElement>> = ({
  ...props
}) => {
  return (
    <svg
      viewBox="0 0 100 100"
      stroke="black"
      strokeWidth={8}
      fill="none"
      {...props}
    >
      <g>
        <path d="M50 0 L20 50 L50 100" />
        <path d="M70 0 L40 50 L70 100" />
      </g>
    </svg>
  );
};

export const Cross: React.FC<SVGAttributes<SVGSVGElement>> = ({ ...props }) => {
  return (
    <svg
      viewBox="0 0 100 100"
      width="100"
      stroke="black"
      strokeWidth={8}
      {...props}
    >
      <line x1="10" y1="10" x2="90" y2="90" />
      <line x1="90" y1="10" x2="10" y2="90" />
    </svg>
  );
};

export const Logo: React.FC<SVGAttributes<SVGSVGElement>> = ({ ...props }) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      xmlnsXlink="http://www.w3.org/1999/xlink"
      id="Layer_1"
      data-name="Layer 1"
      viewBox="0 0 33.36 33.12"
      {...props}
    >
      <g className="cls-1">
        <path
          fill="#eee"
          d="m16.85,4.14c.72,0,1.3.58,1.3,1.3s-.58,1.3-1.3,1.3-1.3-.58-1.3-1.3.58-1.3,1.3-1.3Z"
        />
        <path
          fill="#eee"
          d="m16.86,14.28c.12,0,.24-.03.35-.1l1.42-.91s.56-.38.56-.38c0,0,0-.64,0-.65v-3.39c0-.18.07-.35.19-.47.12-.13.29-.21.49-.21h.85c.59,0,1.15-.27,1.52-.72,0,0,2.6-3.44,2.61-3.43,0,0-.03,2.95-.02,2.95,0,0-2.19,2.78-2.19,2.78-.37.48-.95.76-1.55.76h-.79c-.37,0-.68.31-.68.68v3.78s0,.39,0,.39c0,0-.35.21-.35.21l-2.06,1.18c-.11.06-.24.09-.37.09-.13,0-.25-.03-.37-.09l-2.06-1.18s-.35-.21-.35-.21c0,0,0-.38,0-.39v-3.78c0-.37-.3-.68-.67-.68h-.79c-.61,0-1.18-.28-1.55-.76,0,0-2.18-2.77-2.19-2.78,0,0-.02-2.94-.02-2.95,0,0,2.61,3.43,2.61,3.43.37.46.93.72,1.52.72h.85c.19,0,.36.08.49.21.12.12.19.29.19.47v3.39s0,.65,0,.65c0,0,.55.37.56.38l1.42.91c.11.07.23.1.35.1h0Z"
        />
        <path
          fill="#eee"
          d="m14.41,18.75c-.06-.11-.15-.2-.26-.26l-1.5-.77s-.61-.3-.61-.3c0,0-.55.32-.56.32l-2.93,1.7c-.16.09-.34.11-.51.07-.17-.04-.33-.15-.42-.32l-.43-.74c-.3-.51-.81-.86-1.39-.96,0,0-4.28-.53-4.27-.54,0,0,2.57-1.45,2.57-1.45,0,0,3.5.5,3.5.5.6.08,1.13.44,1.43.96l.4.68c.19.32.6.43.93.25l3.27-1.9s.33-.19.33-.19c0,0,.35.2.36.19l2.05,1.19c.11.07.2.16.26.27.07.11.1.23.1.36v2.37s0,.4,0,.4c0,0-.33.19-.34.19l-3.28,1.88c-.32.19-.44.6-.25.93l.39.68c.3.53.35,1.16.12,1.73,0,0-1.31,3.27-1.31,3.28,0,0-2.54,1.49-2.54,1.5,0,0,1.67-3.97,1.67-3.97.21-.55.16-1.17-.14-1.68l-.43-.74c-.09-.16-.11-.35-.06-.52.05-.16.15-.31.31-.4l2.94-1.69s.56-.33.56-.33c0,0,.05-.66.05-.68l.08-1.68c0-.13-.03-.25-.09-.35h0s0,0,0,0Z"
        />
        <path
          fill="#eee"
          d="m19.31,18.76c-.06.1-.1.23-.09.35l.08,1.68s.05.68.05.68c0,0,.55.32.56.33l2.94,1.69c.16.09.27.24.31.4.05.17.03.36-.06.52l-.43.74c-.29.51-.34,1.13-.14,1.68,0,0,1.68,3.97,1.67,3.97,0,0-2.55-1.5-2.54-1.5,0,0-1.31-3.28-1.31-3.28-.23-.56-.18-1.2.12-1.72l.39-.68c.19-.32.07-.74-.25-.93l-3.28-1.88s-.34-.19-.34-.19c0,0,0-.4,0-.41v-2.37c.01-.13.05-.26.11-.36.06-.11.15-.21.26-.27l2.05-1.19s.35-.19.35-.19c0,0,.33.19.33.19l3.27,1.9c.32.19.74.08.93-.25l.4-.68c.3-.53.83-.88,1.43-.96,0,0,3.49-.5,3.5-.5,0,0,2.56,1.45,2.57,1.45,0,0-4.27.54-4.27.54-.58.1-1.09.45-1.39.96l-.43.74c-.1.16-.25.27-.42.32-.17.04-.35.02-.51-.07l-2.93-1.7s-.56-.32-.56-.32c0,0-.6.29-.61.3l-1.5.77c-.11.06-.2.15-.26.26Z"
        />
      </g>
    </svg>
  );
};
