import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { useEffect, useState } from "react";

// Hook makes sure we don't mount components when the theme is transitioning.
export const useTheme = ({ isDark: targetIsDark }: { isDark: boolean }) => {
  const dispatch = useGlobalContextDispatch();
  const { isThemeTransitioning, isDark } = useGlobalContext();
  const [shouldMount, setShouldMount] = useState(false);

  useEffect(() => {
    if (dispatch && isDark !== targetIsDark) {
      dispatch({ type: "SET_IS_DARK", val: targetIsDark });
      dispatch({ type: "START_THEME_TRANSITION" });
    }
  }, [dispatch, isDark, targetIsDark]);

  useEffect(() => {
    if (isDark === targetIsDark && !isThemeTransitioning) {
      setShouldMount(true);
    }
  }, [isDark, targetIsDark, isThemeTransitioning]);

  return { shouldMount };
};
