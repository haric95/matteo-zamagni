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

export const Rhombus: SelectableIconComponent = ({
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
        style={{ transformOrigin: "50px 50px" }}
        className={`${
          selected
            ? "animate-rhombus_TopRight_Forwards"
            : hasBeenSelected
            ? "animate-rhombus_TopRight_Reverse"
            : ""
        }`}
        {...{ x1: "50", y1: "5", x2: "95", y2: "50" }}
      />
      <line
        style={{ transformOrigin: "50px 50px" }}
        className={`${
          selected
            ? "animate-rhombus_BottomRight_Forwards"
            : hasBeenSelected
            ? "animate-rhombus_BottomRight_Reverse"
            : ""
        }`}
        {...{ x1: "95", y1: "50", x2: "50", y2: "95" }}
      />
      <line
        style={{ transformOrigin: "50px 50px" }}
        className={`${
          selected
            ? "animate-rhombus_BottomLeft_Forwards"
            : hasBeenSelected
            ? "animate-rhombus_BottomLeft_Reverse"
            : ""
        }`}
        {...{ x1: "50", y1: "95", x2: "5", y2: "50" }}
      />
      <line
        style={{ transformOrigin: "50px 50px" }}
        className={`${
          selected
            ? "animate-rhombus_TopLeft_Forwards"
            : hasBeenSelected
            ? "animate-rhombus_TopLeft_Reverse"
            : ""
        }`}
        {...{ x1: "5", y1: "50", x2: "50", y2: "5" }}
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

export const Logo: React.FC<SVGAttributes<SVGSVGElement>> = ({
  color,
  ...props
}) => {
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
          fill={color}
          d="m16.63,3.13c.72,0,1.3.58,1.3,1.3s-.58,1.3-1.3,1.3-1.3-.58-1.3-1.3.58-1.3,1.3-1.3Z"
        />
        <path
          fill={color}
          d="m23.7,29.18l-.24-.45c-.1-.19-.2-.39-.27-.6l-.9-2.4c-.21-.55-.15-1.17.14-1.68l.43-.74c.19-.32.08-.74-.25-.93l-2.93-1.7c-.36-.21-.59-.59-.61-1.01l-.07-1.68c-.01-.24.12-.46.33-.57l1.5-.77c.37-.19.82-.18,1.18.03l2.93,1.7c.32.19.74.08.93-.25l.43-.74c.3-.51.81-.86,1.39-.96l2.53-.42c.22-.04.44-.06.66-.06h.51c.17-.02.23-.25.08-.34l-1.48-.86c-.35-.2-.76-.28-1.16-.23l-2.92.41c-.6.08-1.13.44-1.43.96l-.4.68c-.19.32-.6.43-.93.25l-3.27-1.9c-.21-.12-.48-.12-.69,0l-2.05,1.19c-.21.12-.34.35-.34.59l-.02,2.37c0,.25.13.48.34.6l3.27,1.89c.32.19.43.6.25.93l-.4.68c-.3.53-.35,1.16-.12,1.72l1.1,2.74c.15.38.42.69.77.89l1.48.86c.15.09.32-.08.24-.23"
        />
        <path
          fill={color}
          d="m1.87,15.89h.51c.22.02.44.04.66.08l2.53.42c.58.1,1.09.45,1.38.96l.43.74c.19.32.6.44.93.25l2.94-1.69c.36-.21.81-.22,1.18-.02l1.49.78c.21.11.34.33.33.57l-.08,1.68c-.02.42-.25.8-.61,1l-2.94,1.69c-.32.19-.44.6-.25.93l.43.74c.29.51.34,1.13.14,1.68l-.9,2.4c-.08.21-.17.41-.27.6l-.24.45c-.08.15.09.32.24.23l1.49-.85c.35-.2.62-.52.78-.89l1.11-2.73c.23-.56.18-1.2-.12-1.73l-.39-.68c-.19-.32-.07-.74.25-.93l3.28-1.88c.21-.12.35-.35.34-.6v-2.37c-.01-.24-.14-.47-.35-.59l-2.04-1.2c-.21-.12-.48-.13-.69,0l-3.28,1.88c-.32.19-.74.07-.93-.25l-.39-.68c-.3-.53-.83-.88-1.43-.97l-2.92-.42c-.4-.06-.81.02-1.16.22l-1.49.85c-.15.09-.09.32.08.32"
        />
        <path
          fill={color}
          d="m24.29,3.69l-.27.43c-.12.19-.24.37-.38.54l-1.63,1.98c-.38.46-.94.72-1.53.72h-.85c-.37,0-.68.3-.68.68v3.39c0,.42-.22.81-.57,1.03l-1.42.9c-.2.13-.46.13-.66,0l-1.42-.91c-.35-.23-.56-.62-.56-1.03v-3.39c0-.37-.3-.68-.67-.68h-.85c-.59,0-1.15-.27-1.52-.72l-1.62-1.98c-.14-.17-.27-.35-.38-.54l-.27-.43c-.09-.15-.32-.08-.32.09v1.71c0,.41.13.8.38,1.12l1.81,2.33c.37.48.95.76,1.55.76h.79c.37,0,.68.31.68.68v3.78c0,.25.12.47.34.6l2.06,1.18c.21.12.47.12.68,0l2.06-1.17c.21-.12.35-.35.35-.6v-3.78c0-.37.31-.68.69-.68h.79c.61,0,1.18-.28,1.56-.75l1.82-2.32c.25-.32.39-.71.39-1.12v-1.71c0-.17-.23-.24-.32-.09"
        />
      </g>
    </svg>
  );
};
