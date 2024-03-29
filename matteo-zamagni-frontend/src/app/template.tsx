"use client";
import { useGlobalContext } from "@/state/GlobalStore";
import { motion } from "framer-motion";
import { PropsWithChildren } from "react";

const variants = {
  hidden: { opacity: 0, x: 0, y: 0 },
  enter: { opacity: 1, x: 0, y: 0 },
};

const Template: React.FC<PropsWithChildren> = ({ children }) => {
  const { gridDim } = useGlobalContext();

  return (
    <motion.main
      className={`w-full h-full absolute grid`}
      style={{
        gridTemplateColumns: `repeat(${gridDim?.x}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${gridDim?.y}, minmax(0, 1fr))`,
      }}
      variants={variants}
      initial={{ opacity: 0 }}
      exit={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ type: "linear", duration: 0.5 }}
    >
      {children}
    </motion.main>
  );
};

export default Template;
