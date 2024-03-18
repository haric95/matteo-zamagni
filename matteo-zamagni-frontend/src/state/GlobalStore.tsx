"use client";

import { ScreenDim } from "@/types/global";
import {
  Dispatch,
  PropsWithChildren,
  Reducer,
  createContext,
  useContext,
  useReducer,
} from "react";

type GlobalState = { screenDim: ScreenDim | null };
type GlobalDispatch = Dispatch<GlobalAction> | null;

const initialGlobalState: GlobalState = { screenDim: null };
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

type GlobalAction = { type: "SET_DIM"; dim: ScreenDim };

const globalReducer: Reducer<GlobalState, GlobalAction> = (
  globalState,
  action
) => {
  switch (action.type) {
    case "SET_DIM": {
      console.log(action.dim);
      return { ...globalState, screenDim: action.dim };
    }
    default: {
      throw Error("Unknown action: " + action.type);
    }
  }
};
