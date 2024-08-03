"use client";
import { TARGET_CELL_SIZE } from "@/hooks/useScreenDim";
import { useGlobalContext } from "@/state/GlobalStore";
import { motion } from "framer-motion";
import { PropsWithChildren } from "react";
import "../helpers/polyfills";

const variants = {
  hidden: { opacity: 0, x: 0, y: 0 },
  enter: { opacity: 1, x: 0, y: 0 },
};

const Template: React.FC<PropsWithChildren> = ({ children }) => {
  const { gridDim } = useGlobalContext();

  return (
    <motion.main
      className={`absolute grid`}
      style={{
        gridTemplateColumns: `repeat(${gridDim?.x}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${gridDim?.y}, minmax(0, 1fr))`,
        width: gridDim ? `${gridDim.x * TARGET_CELL_SIZE}px` : "100%",
        height: gridDim ? `${gridDim.y * TARGET_CELL_SIZE}px` : "100%",
      }}
      // variants={variants}
      initial={{ opacity: 0 }}
      exit={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ type: "ease-in-out", duration: 0.5 }}
    >
      {children}
    </motion.main>
  );
};

export default Template;
