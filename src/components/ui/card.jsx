import React from "react";

function joinClasses(...classes) {
  return classes.filter(Boolean).join(" ");
}

export const Card = React.forwardRef(function Card({ className = "", ...props }, ref) {
  return <div ref={ref} className={joinClasses("rounded-xl border bg-neutral-900", className)} {...props} />;
});

export const CardContent = React.forwardRef(function CardContent({ className = "", ...props }, ref) {
  return <div ref={ref} className={joinClasses("p-6", className)} {...props} />;
});
