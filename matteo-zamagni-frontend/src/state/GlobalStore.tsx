"use client";

import { clearGrid } from "@/helpers/gridHelpers";
import { Dim2D, Grid, Pos2D } from "@/types/global";
import {
  Dispatch,
  PropsWithChildren,
  Reducer,
  createContext,
  useContext,
  useReducer,
} from "react";

type GlobalState = {
  screenDim: Dim2D | null;
  gridDim: Dim2D | null;
  grid: boolean[][] | null;
  isDark: boolean;
  selectedYear: null | string;
  hasLoaded: boolean;
  isThemeTransitioning: boolean;
  creditsIsOpen: boolean;
  cellSize: Dim2D | null;
  mobileFooterMenuOpen: boolean;
  scrollerAvailableYears: string[] | null;
};
type GlobalDispatch = Dispatch<GlobalAction> | null;

const initialGlobalState: GlobalState = {
  screenDim: null,
  gridDim: null,
  grid: null,
  isDark: true,
  selectedYear: null,
  hasLoaded: false,
  isThemeTransitioning: false,
  creditsIsOpen: false,
  cellSize: null,
  mobileFooterMenuOpen: false,
  scrollerAvailableYears: null,
};
const initialGlobalDispatchState: GlobalDispatch = null;

const GlobalContext = createContext<GlobalState>(initialGlobalState);

const GlobalDispatchContext = createContext<GlobalDispatch>(
  initialGlobalDispatchState
);

export const GlobalContextProvider: React.FC<PropsWithChildren> = ({
  children,
}) => {
  const [globalState, dispatch] = useReducer(globalReducer, initialGlobalState);

  return (
    <GlobalContext.Provider value={globalState}>
      <GlobalDispatchContext.Provider value={dispatch}>
        {children}
      </GlobalDispatchContext.Provider>
    </GlobalContext.Provider>
  );
};

export function useGlobalContext() {
  return useContext(GlobalContext);
}

export function useGlobalContextDispatch() {
  return useContext(GlobalDispatchContext);
}

type GlobalAction =
  | { type: "SET_SCREEN_DIM"; dim: Dim2D }
  | { type: "SET_GRID_DIM"; dim: Dim2D }
  | { type: "UPDATE_GRID"; grid: Grid }
  | { type: "CLEAR_GRID" }
  | { type: "SET_IS_DARK"; val: boolean }
  | { type: "SET_LED_ANIMATION_START" }
  | { type: "SET_LED_ANIMATION_END" }
  | { type: "SET_SELECTED_YEAR"; year: GlobalState["selectedYear"] }
  | { type: "START_THEME_TRANSITION" }
  | { type: "END_THEME_TRANSITION" }
  | { type: "OPEN_CREDITS" }
  | { type: "CLOSE_CREDITS" }
  | { type: "SET_LOADED" }
  | { type: "SET_CELL_SIZE"; dim: Pos2D }
  | { type: "SET_MOBILE_FOOTER_MENU"; isOpen: boolean }
  | { type: "SET_SCROLLER_AVAILABLE_YEARS"; years: string[] };

const globalReducer: Reducer<GlobalState, GlobalAction> = (
  globalState,
  action
) => {
  switch (action.type) {
    case "SET_SCREEN_DIM": {
      return { ...globalState, screenDim: action.dim };
    }
    case "SET_GRID_DIM": {
      return {
        ...globalState,
        gridDim: action.dim,
        grid: new Array(action.dim.y).fill(new Array(action.dim.x).fill(false)),
      };
    }
    case "UPDATE_GRID": {
      return { ...globalState, grid: action.grid };
    }
    case "CLEAR_GRID": {
      return {
        ...globalState,
        grid: globalState.grid ? clearGrid(globalState.grid) : null,
      };
    }
    case "SET_IS_DARK": {
      return {
        ...globalState,
        isDark: action.val,
      };
    }
    case "SET_SELECTED_YEAR": {
      return { ...globalState, selectedYear: action.year };
    }
    case "SET_LOADED": {
      return { ...globalState, hasLoaded: true };
    }
    case "START_THEME_TRANSITION": {
      return { ...globalState, isThemeTransitioning: true };
    }
    case "END_THEME_TRANSITION": {
      return { ...globalState, isThemeTransitioning: false };
    }
    case "OPEN_CREDITS": {
      return { ...globalState, creditsIsOpen: true };
    }
    case "CLOSE_CREDITS": {
      return { ...globalState, creditsIsOpen: false };
    }
    case "SET_CELL_SIZE": {
      return { ...globalState, cellSize: action.dim };
    }
    case "SET_MOBILE_FOOTER_MENU": {
      return { ...globalState, mobileFooterMenuOpen: action.isOpen };
    }
    case "SET_SCROLLER_AVAILABLE_YEARS": {
      return { ...globalState, scrollerAvailableYears: action.years };
    }
    default: {
      throw Error("Unknown action: " + JSON.stringify(action));
    }
  }
};
