import { SVGAttributes } from "react";

export const Plus: React.FC<SVGAttributes<SVGSVGElement>> = ({ ...props }) => {
  return (
    <svg viewBox="0 0 100 100" width="100" {...props}>
      <line x1="50" y1="10" x2="50" y2="90" />
      <line x1="10" y1="50" x2="90" y2="50" />
    </svg>
  );
};

export const TriangleDown: React.FC<SVGAttributes<SVGSVGElement>> = ({
  ...props
}) => {
  return (
    <svg viewBox="0 0 100 100" fill="none" stroke="black" {...props}>
      <path d="M10 15 L90 15 L50 85 Z" />
    </svg>
  );
};

export const HorizontalLines: React.FC<SVGAttributes<SVGSVGElement>> = ({
  ...props
}) => {
  return (
    <svg viewBox="0 0 100 100" stroke="black" {...props}>
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
    <svg viewBox="0 0 100 100" fill="none" stroke="black" {...props}>
      <circle cx="50" cy="50" r="45" />
    </svg>
  );
};

export const Star: React.FC<SVGAttributes<SVGSVGElement>> = ({ ...props }) => {
  return (
    <svg viewBox="0 0 100 100" stroke="black" fill="none" {...props}>
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
    <svg viewBox="0 0 100 100" stroke="black" fill="none" {...props}>
      <g>
        <path d="M50 0 L20 50 L50 100" />
        <path d="M70 0 L40 50 L70 100" />
      </g>
    </svg>
  );
};
