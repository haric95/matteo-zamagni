"use client";

import { usePathname } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import FrozenRoute from "./FrozenRoute";
import { PropsWithChildren } from "react";
import { useGlobalContext } from "@/state/GlobalStore";
import { TARGET_CELL_SIZE } from "@/hooks/useScreenDim";

const PageAnimatePresence: React.FC<PropsWithChildren> = ({ children }) => {
  const pathname = usePathname();
  const { gridDim } = useGlobalContext();

  return (
    <AnimatePresence mode="wait">
      {/**
       * We use `motion.div` as the first child of `<AnimatePresence />` Component so we can specify page animations at the page level.
       * The `motion.div` Component gets re-evaluated when the `key` prop updates, triggering the animation's lifecycles.
       * During this re-evaluation, the `<FrozenRoute />` Component also gets updated with the new route components.
       */}
      <motion.div
        className={`absolute grid`}
        style={{
          gridTemplateColumns: `repeat(${gridDim?.x}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${gridDim?.y}, minmax(0, 1fr))`,
          width: gridDim ? `${gridDim.x * TARGET_CELL_SIZE}px` : "100%",
          height: gridDim ? `${gridDim.y * TARGET_CELL_SIZE}px` : "100%",
        }}
        key={pathname}
      >
        <FrozenRoute>{children}</FrozenRoute>
      </motion.div>
    </AnimatePresence>
  );
};

export default PageAnimatePresence;
