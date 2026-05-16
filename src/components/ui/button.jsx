import React from "react";

function joinClasses(...classes) {
  return classes.filter(Boolean).join(" ");
}

const variants = {
  default: "bg-white text-neutral-950 hover:bg-neutral-200",
  secondary: "bg-neutral-800 text-white hover:bg-neutral-700",
  outline: "border border-neutral-700 bg-transparent text-white hover:bg-neutral-800"
};

export const Button = React.forwardRef(function Button(
  { className = "", variant = "default", type = "button", ...props },
  ref
) {
  return (
    <button
      ref={ref}
      type={type}
      className={joinClasses(
        "inline-flex items-center justify-center whitespace-nowrap rounded-md px-4 py-2 text-sm font-medium transition-colors disabled:pointer-events-none disabled:opacity-50",
        variants[variant] || variants.default,
        className
      )}
      {...props}
    />
  );
});
