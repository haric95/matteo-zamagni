import { SVGAttributes, useEffect, useState } from "react";

export type IconComponent = React.FC<SVGAttributes<SVGSVGElement>>;

export type SelectableIconComponent = React.FC<
  SVGAttributes<SVGSVGElement> & { selected: boolean }
>;

export const Plus: SelectableIconComponent = ({ selected, ...props }) => {
  return (
    <svg
      viewBox="0 0 100 100"
      width="100"
      stroke={"black"}
      strokeWidth={8}
      className={`transition-all duration-500 ${selected ? "rotate-45" : ""}`}
      {...props}
    >
      <line {...{ x1: "50", y1: "10", x2: "50", y2: "90" }} />
      <line {...{ x1: "10", y1: "50", x2: "90", y2: "50" }} />
    </svg>
  );
};

export const TriangleDown: SelectableIconComponent = ({
  selected,
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
